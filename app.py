import os
import uuid
import tempfile
import subprocess
from flask import Flask, render_template, request, redirect, url_for, send_file, flash, abort
import fitz  # PyMuPDF
import hashlib  # for detecting repeated images
import numpy as np  # for image processing
import cv2  # OpenCV for watermark detection and removal


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

    # Determine if PDF contains text; if not, run OCR
    if not pdf_has_text(input_path):
        input_path = run_ocr(input_path)

    # Extract images and create a new PDF, optionally removing watermark
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
    # Enable debug mode for local development
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)