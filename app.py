import os
import uuid
import tempfile
import subprocess
from flask import Flask, render_template, request, redirect, url_for, send_file, flash, abort
import fitz  # PyMuPDF
import hashlib  # for detecting repeated images
import numpy as np  # for image processing
import cv2  # OpenCV for watermark detection and removal
import threading  # for background tasks
import urllib.request  # for pinging external URLs
import time  # for sleep intervals


"""
This Flask application implements a simple web interface that accepts a PDF from
the user, analyses it for textual content and images, and returns a new PDF
containing only the images found on the original pages. If the uploaded PDF
contains no selectable text (i.e. it's a scanned document), the app
automatically attempts to run OCR on it using OCRmyPDF to make it
searchable before processing.  

Workflow:

1. A user uploads a PDF via the form on the index page.
2. The server saves the file to a temporary location.
3. The PDF is opened with PyMuPDF (`fitz`) and checked for textual content on any page.
4. If no text is found, the file is passed through OCRmyPDF (`ocrmypdf`) to
   produce a searchable PDF.  This step uses the `--force-ocr` and
   `--output-type pdf` options to always run OCR and output a standard PDF.
5. The server iterates through each page, extracts every image block (block
   type 1) and inserts it into a new blank page of identical size in a new
   document.  Other block types (text, drawings, watermarks) are ignored so
   that only images remain on the page.
6. The new PDF is saved to a temporary file.  A unique ID is generated for
   this file and stored in memory so that the user can download it via a
   simple link.

The app exposes three routes:

* `/` – shows the upload form.
* `/process` – handles the uploaded file and creates the processed PDF.
* `/download/<file_id>` – serves the processed PDF for download.

Note: The use of OCRmyPDF requires that the `ocrmypdf` CLI be installed on
the deployment environment.  If OCR fails for any reason (for example,
OCRmyPDF is not installed), the app will continue processing the original
document and simply extract images from it.

"""

import os

# Determine the directory containing this file.  Use it to locate the
# templates directory explicitly.  This makes it possible to run the app
# even if the working directory changes or the app is deployed from a
# different location.  Flask defaults to looking for a ``templates``
# directory relative to the current working directory; specifying
# ``template_folder`` ensures the correct path is used.
basedir = os.path.abspath(os.path.dirname(__file__))
primary_template_dir = os.path.join(basedir, 'templates')
fallback_template_dir = os.path.join(basedir, 'sharepremium', 'templates')

# Determine which template directory actually contains index.html.  This
# fallback logic allows the application to run correctly regardless of
# whether the repository is structured with templates/ at the top level
# or nested inside a "sharepremium" directory (for example, if the
# project was zipped with an extra folder).  If neither directory
# contains the expected template, Flask will raise an error as usual.
if os.path.exists(os.path.join(primary_template_dir, 'index.html')):
    chosen_template_dir = primary_template_dir
elif os.path.exists(os.path.join(fallback_template_dir, 'index.html')):
    chosen_template_dir = fallback_template_dir
else:
    # Default to the primary directory; errors will surface later if
    # templates are indeed missing.
    chosen_template_dir = primary_template_dir

app = Flask(__name__, template_folder=chosen_template_dir)
# A secret key is required for flash messages; this should be changed in
# production via an environment variable.
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change-me")

# In‑memory store mapping unique IDs to processed file paths.  In a more
# robust deployment you might prefer a database or persistent storage.
processed_files = {}

# ---------------------------------------------------------------------------
# Background ping functionality

def ping_google_loop():
    """Continuously send an HTTP GET request to google.com every 60 seconds.

    This function runs in a separate daemon thread. If any network error
    occurs (for example, no internet access), the exception is silently
    ignored and the loop continues. Using an HTTP request instead of the
    system `ping` command avoids platform dependencies and does not
    require external binaries.
    """
    while True:
        try:
            urllib.request.urlopen('https://testwebapp-1k7i.onrender.com', timeout=10)
        except Exception:
            pass
        time.sleep(60)

def start_ping_thread():
    """Start the background ping thread if it hasn't been started yet."""
    if not hasattr(start_ping_thread, '_thread_started'):
        thread = threading.Thread(target=ping_google_loop, daemon=True)
        thread.start()
        start_ping_thread._thread_started = True

# Register a handler to start the ping thread before the first request. In
# production (e.g., under Gunicorn), this ensures each worker process
# initiates the background thread exactly once. For local development,
# the thread will also start when the app is served.
@app.before_first_request
def _start_background_tasks():
    start_ping_thread()


def pdf_has_text(pdf_path: str) -> bool:
    """Return True if at least one page of the PDF has selectable text.

    The function opens the PDF with PyMuPDF and iterates through each page,
    retrieving plain text via `page.get_text()`.  If any non‑empty string is
    returned, the PDF is considered to be searchable (i.e. not purely
    scanned).
    """
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text = page.get_text().strip()
                if text:
                    return True
    except Exception:
        # In case of errors reading the PDF, assume no text to force OCR
        return False
    return False


def run_ocr(input_path: str) -> str:
    """Run OCR on the given PDF using OCRmyPDF.

    This function invokes the `ocrmypdf` command line tool via subprocess.
    The `--force-ocr` option ensures that OCR runs even if the file appears
    to already contain text.  The `--output-type pdf` argument produces a
    regular PDF rather than a PDF/A.  A new temporary file is created for
    the OCR result.

    Returns the path to the OCR‑processed PDF.  If OCR fails, the
    original input path is returned.
    """
    # Create a temporary file for the OCR output
    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    tmp_out.close()
    ocr_output_path = tmp_out.name

    try:
        # Build the ocrmypdf command
        cmd = [
            'ocrmypdf',
            '--force-ocr',  # always run OCR even if text exists
            '--output-type', 'pdf',
            input_path,
            ocr_output_path
        ]
        # Run the command, suppressing output to keep logs clean
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return ocr_output_path
    except Exception:
        # If OCR fails (for example, ocrmypdf isn't installed), return the original
        # input path so the app continues processing without OCR
        try:
            os.unlink(ocr_output_path)
        except OSError:
            pass
        return input_path


def extract_images_only(input_path: str, watermark_pattern: 'np.ndarray | None' = None) -> str:
    """Create a new PDF containing only the images from the input PDF.

    Opens the input PDF with PyMuPDF (`fitz`) and iterates through its
    pages.  For each page, a corresponding blank page of the same size is
    created in a new document.  The function then inspects the page's
    content via `get_text("dict")` to find blocks of type 1, which
    represent embedded images (according to the PyMuPDF documentation
    describing image blocks【740116005133473†L274-L297】).  For each image block,
    the image bytes and bounding box are retrieved and inserted into the
    new page using `insert_image`.  All other content (text, vector
    graphics, watermarks) is ignored.

    Returns the path to the newly created PDF.
    """
    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    tmp_out.close()
    output_path = tmp_out.name
    
    with fitz.open(input_path) as doc:
        # ------------------------------------------------------------------
        # First pass: count occurrences of each image across pages.  Images
        # that appear on a large fraction of pages are likely watermarks (e.g.,
        # repeated logos or headers).  We compute an MD5 hash of each image
        # to identify duplicates.  This pass does not modify the document.
        image_counts = {}
        for page in doc:
            try:
                blocks = page.get_text("dict").get("blocks", [])
            except Exception:
                blocks = []
            for block in blocks:
                if block.get('type') == 1:
                    img_bytes = block.get('image')
                    if not img_bytes:
                        continue
                    md5 = hashlib.md5(img_bytes).hexdigest()
                    image_counts[md5] = image_counts.get(md5, 0) + 1
        num_pages = max(doc.page_count, 1)
        # Determine which image hashes should be treated as watermarks based on
        # their frequency.  Images appearing on a large fraction of pages are
        # likely to be repeated logos or headers.  The default threshold (0.8)
        # means the image must appear on at least 80% of pages to be considered
        # a watermark.  This threshold can be overridden via the
        # WATERMARK_FREQ_THRESHOLD environment variable.
        try:
            freq_threshold = float(os.environ.get("WATERMARK_FREQ_THRESHOLD", 0.8))
        except Exception:
            freq_threshold = 0.8
        watermark_hashes = set()
        for md5_hash, count in image_counts.items():
            if count / num_pages >= freq_threshold:
                watermark_hashes.add(md5_hash)

        # ------------------------------------------------------------------
        # Second pass: construct the output PDF.  We create a new blank page
        # corresponding to each original page and insert only those images
        # that are not too large (background) and not identified as watermarks.
        out_doc = fitz.open()
        for page in doc:
            new_page = out_doc.new_page(width=page.rect.width, height=page.rect.height)
            try:
                blocks = page.get_text("dict").get("blocks", [])
            except Exception:
                blocks = []
            page_area = page.rect.width * page.rect.height
            try:
                max_ratio = float(os.environ.get("MAX_IMAGE_AREA_RATIO", 0.2))
            except Exception:
                max_ratio = 0.2
            for block in blocks:
                if block.get('type') != 1:
                    continue
                bbox = block.get('bbox')
                img_bytes = block.get('image')
                if not bbox or not img_bytes:
                    continue
                md5 = hashlib.md5(img_bytes).hexdigest()
                # Skip images identified as watermarks based on frequency
                if md5 in watermark_hashes:
                    continue
                # Compute area ratio to skip large background images
                rect_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                ratio = 0.0
                if page_area:
                    try:
                        ratio = rect_area / page_area
                    except Exception:
                        ratio = 0.0
                if ratio > max_ratio:
                    continue
                # Optionally remove user‑provided watermark from the image
                processed_bytes = img_bytes
                if watermark_pattern is not None:
                    try:
                        processed_bytes = remove_watermark_from_image(img_bytes, watermark_pattern)
                    except Exception:
                        # If removal fails, fall back to original bytes
                        processed_bytes = img_bytes
                rect = fitz.Rect(bbox)
                try:
                    new_page.insert_image(rect, stream=processed_bytes)
                except Exception:
                    pass
        # Save and close
        out_doc.save(output_path)
        out_doc.close()
    return output_path


def remove_watermarks_only(input_path: str, watermark_pattern: 'np.ndarray | None' = None) -> str:
    """Create a new PDF by removing watermark images but keeping all text and other content.

    This function opens the input PDF and iterates through its pages, identifying
    embedded images via their cross‑reference numbers (xrefs).  It applies three
    heuristics to decide whether an image should be removed:

    * **Repeated logos** – Images that appear on a large fraction of pages are
      treated as watermarks.  A first pass computes an MD5 digest for every
      image and counts how often it occurs.  Any digest whose frequency
      meets or exceeds the fraction defined by the environment variable
      ``WATERMARK_FREQ_THRESHOLD`` (default 0.8) is considered a watermark.
    * **Page‑wide backgrounds** – Images whose area covers more than a
      fraction of the page (controlled via ``MAX_IMAGE_AREA_RATIO``, default
      0.2) are skipped.  These are often decorative page backgrounds or
      watermarks printed across the entire page.
    * **User‑supplied pattern** – If the user uploads a sample watermark
      image, the function uses OpenCV template matching to locate that
      pattern within each embedded image.  When a match exceeds the
      correlation threshold (``WATERMARK_MATCH_THRESHOLD``, default 0.8), a
      cleaned version of the image is created by inpainting the matched
      region, and inserted over the original bounding box.  The original
      image is not deleted in this case – the overlay simply hides the
      watermark portion.

    Images identified as watermarks by the first two heuristics are removed
    entirely using ``page.delete_image(xref)``, which globally replaces the
    image with a transparent 1×1 pixel【8857088909701†L2286-L2297】.  In contrast,
    images cleaned via the pattern matching heuristic are left intact and
    overlaid with the cleaned version.

    Args:
        input_path: Path to the original PDF.
        watermark_pattern: Optional grayscale NumPy array containing a
            sample watermark to detect and remove via template matching.

    Returns:
        Path to the processed PDF with watermarks removed and text preserved.
    """
    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    tmp_out.close()
    output_path = tmp_out.name

    with fitz.open(input_path) as doc:
        # ------------------------------------------------------------------
        # First pass: compute MD5 hashes of all embedded images and count
        # occurrences across pages.  Duplicates appearing on many pages are
        # considered watermarks.
        image_counts = {}
        xrefs_replaced = set()
        for page in doc:
            # Use get_image_info to retrieve image xrefs and (optionally) hashes
            try:
                info_list = page.get_image_info(hashes=True, xrefs=True)
            except Exception:
                info_list = []
            for info in info_list:
                digest = info.get('digest')
                if not digest:
                    # fallback: compute digest via extract_image
                    xref = info.get('xref')
                    if xref:
                        try:
                            img_bytes = doc.extract_image(xref).get('image')
                        except Exception:
                            img_bytes = None
                        if img_bytes:
                            digest = hashlib.md5(img_bytes).digest()
                if digest:
                    image_counts[digest] = image_counts.get(digest, 0) + 1
        # Determine frequency threshold
        num_pages = max(doc.page_count, 1)
        try:
            freq_threshold = float(os.environ.get("WATERMARK_FREQ_THRESHOLD", 0.8))
        except Exception:
            freq_threshold = 0.8
        watermark_digests = set(digest for digest, count in image_counts.items() if count / num_pages >= freq_threshold)

        # Keep track of xrefs that should be deleted across the document
        xrefs_to_delete = set()

        # ------------------------------------------------------------------
        # Second pass: iterate pages and handle each image according to
        # heuristics.  For deletions we collect xrefs; for pattern matching
        # overlays we insert cleaned images directly into the page.
        for page in doc:
            page_area = page.rect.width * page.rect.height
            try:
                max_ratio = float(os.environ.get("MAX_IMAGE_AREA_RATIO", 0.2))
            except Exception:
                max_ratio = 0.2
            # Retrieve image info with xrefs and bounding boxes
            try:
                images = page.get_images(full=True)
            except Exception:
                images = []
            for img in images:
                # Each item: (xref, smask, width, height, bpc, colorspace, alt, name, filter)
                xref = img[0]
                if not xref:
                    continue
                # Extract digest for this image
                try:
                    img_bytes = doc.extract_image(xref).get('image')
                except Exception:
                    img_bytes = None
                digest = None
                if img_bytes:
                    digest = hashlib.md5(img_bytes).digest()
                # Evaluate repeated logo watermark
                if digest and digest in watermark_digests:
                    xrefs_to_delete.add(xref)
                    continue
                # Evaluate area ratio (background detection).  There may be multiple
                # occurrences of the same xref on a page; get rects for each.
                try:
                    rects = page.get_image_rects(xref)
                except Exception:
                    rects = []
                remove_due_to_area = False
                for rect in rects:
                    rect_area = rect.width * rect.height
                    ratio = 0.0
                    if page_area:
                        try:
                            ratio = rect_area / page_area
                        except Exception:
                            ratio = 0.0
                    if ratio > max_ratio:
                        remove_due_to_area = True
                        break
                if remove_due_to_area:
                    xrefs_to_delete.add(xref)
                    continue
                # If a watermark pattern is provided, try to remove it via inpainting
                if watermark_pattern is not None and img_bytes:
                    try:
                        # Remove the watermark from the image using template matching
                        cleaned_bytes = remove_watermark_from_image(img_bytes, watermark_pattern)
                        # If a change occurred and we haven't replaced this xref yet, replace it globally
                        if cleaned_bytes != img_bytes and xref not in xrefs_replaced:
                            try:
                                # Replace the image at xref globally with the cleaned version. This
                                # ensures the watermark is removed rather than overlaid【8857088909701†L2286-L2297】.
                                page.replace_image(xref, stream=cleaned_bytes)
                                xrefs_replaced.add(xref)
                            except Exception:
                                pass
                    except Exception:
                        pass
        # Apply deletions globally
        for xref in xrefs_to_delete:
            try:
                # page.delete_image is global and may be invoked on any page
                doc[0].delete_image(xref)
            except Exception:
                try:
                    # fallback: call on current page or document
                    page.delete_image(xref)  # type: ignore
                except Exception:
                    pass
        # Save the modified document
        doc.save(output_path)
    return output_path


# ---------------------------------------------------------------------------
# Image watermark removal helper
def remove_watermark_from_image(img_bytes: bytes, pattern: np.ndarray, match_threshold: float = None) -> bytes:
    """Attempt to remove a watermark from an image by template matching and inpainting.

    This helper uses OpenCV to locate occurrences of the user‑supplied watermark
    pattern within the image.  If the pattern is found with correlation above
    the specified threshold, a mask is created over all matched regions and
    inpainting is applied to those regions to erase the watermark.  The
    resulting image is returned as PNG bytes.  If no match is found or
    processing fails, the original image bytes are returned.

    Args:
        img_bytes: Raw bytes of the encoded image (e.g. JPEG or PNG).
        pattern: Grayscale numpy array representing the watermark pattern.
        match_threshold: Minimum normalized correlation (0.0 to 1.0) for a match.  If
            None, the value of the WATERMARK_MATCH_THRESHOLD environment
            variable is used, defaulting to 0.8.

    Returns:
        Bytes of the processed image, encoded as PNG.
    """
    # Determine threshold from environment if not provided
    if match_threshold is None:
        try:
            match_threshold = float(os.environ.get("WATERMARK_MATCH_THRESHOLD", 0.8))
        except Exception:
            match_threshold = 0.8
    # Decode the image bytes to a BGR array
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None or pattern is None:
        return img_bytes
    # Convert to grayscale for matching
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pat_gray = pattern
    # Ensure pattern is smaller than image
    if pat_gray.shape[0] > img_gray.shape[0] or pat_gray.shape[1] > img_gray.shape[1]:
        return img_bytes
    # Perform template matching
    res = cv2.matchTemplate(img_gray, pat_gray, cv2.TM_CCOEFF_NORMED)
    # Identify all locations above threshold
    loc = np.where(res >= match_threshold)
    if loc[0].size == 0:
        # No match: return original
        return img_bytes
    # Create a mask for inpainting
    mask = np.zeros(img_gray.shape, dtype=np.uint8)
    h, w = pat_gray.shape
    # ``loc`` returns (y, x) arrays; zip reversed to get (x, y)
    for pt_y, pt_x in zip(*loc):
        # Bound within image
        y1 = int(pt_y)
        y2 = int(min(pt_y + h, mask.shape[0]))
        x1 = int(pt_x)
        x2 = int(min(pt_x + w, mask.shape[1]))
        mask[y1:y2, x1:x2] = 255
    # Inpaint the watermark areas
    inpaint_radius = 3
    try:
        inpainted = cv2.inpaint(img, mask, inpaint_radius, flags=cv2.INPAINT_TELEA)
    except Exception:
        return img_bytes
    # Encode to PNG
    success, encoded = cv2.imencode('.png', inpainted)
    if not success:
        return img_bytes
    return encoded.tobytes()


@app.route('/', methods=['GET'])
def index():
    """Render the upload form."""
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process_pdf():
    """Handle the uploaded PDF and create a new PDF containing only images."""
    if 'pdf_file' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))
    pdf_file = request.files['pdf_file']
    if pdf_file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))
    # Only accept PDF files based on extension
    if not pdf_file or not pdf_file.filename.lower().endswith('.pdf'):
        flash('Please upload a PDF file')
        return redirect(url_for('index'))

    # Optional watermark file
    watermark_np = None
    w_file = request.files.get('watermark_file')
    if w_file and w_file.filename:
        # Save watermark temporarily and load as grayscale
        tmp_w = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(w_file.filename)[1])
        w_file.save(tmp_w.name)
        tmp_w.close()
        try:
            import cv2
            import numpy as np
            # Read watermark as grayscale
            arr = np.fromfile(tmp_w.name, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                watermark_np = img
        except Exception:
            watermark_np = None
    # Save the uploaded PDF to a temporary location
    tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    pdf_file.save(tmp_in.name)
    input_path = tmp_in.name

    # Determine if the user selected a removal mode: remove both text and watermarks,
    # or remove only watermarks while keeping text.  Default is "both".
    remove_mode = request.form.get('remove_mode', 'both')

    # If we will preserve text (watermark_only), ensure the PDF is searchable.  If
    # the PDF has no selectable text, attempt to run OCR to add a text layer.  If
    # OCR fails, we proceed with the original document.
    if remove_mode == 'watermark_only':
        if not pdf_has_text(input_path):
            input_path = run_ocr(input_path)
        # Remove watermark images while keeping text and other content
        output_path = remove_watermarks_only(input_path, watermark_pattern=watermark_np)
    else:
        # For the default case (both), run OCR if the PDF has no text
        if not pdf_has_text(input_path):
            input_path = run_ocr(input_path)
        # Extract images only (discard text and other content)
        output_path = extract_images_only(input_path, watermark_pattern=watermark_np)

    # Register the processed file in the in‑memory store
    file_id = str(uuid.uuid4())
    processed_files[file_id] = output_path

    return render_template('result.html', file_id=file_id)


@app.route('/download/<file_id>')
def download_file(file_id: str):
    """Serve the processed PDF for download."""
    path = processed_files.get(file_id)
    if path and os.path.exists(path):
        # Use a generic filename for the download
        return send_file(path, as_attachment=True, download_name='images_only.pdf')
    # If the file is not found, return 404
    abort(404)


if __name__ == '__main__':
    # When running the app directly (e.g., via `python app.py`), start the
    # background ping thread explicitly. In production under Gunicorn, the
    # thread will be started by the `before_first_request` handler above.
    start_ping_thread()

    # Enable debug mode for local development
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)
