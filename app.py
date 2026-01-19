import os
import uuid
import tempfile
import subprocess
from flask import Flask, render_template, request, redirect, url_for, send_file, flash, abort
import fitz  # PyMuPDF


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

app = Flask(__name__)
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


def extract_images_only(input_path: str) -> str:
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
        out_doc = fitz.open()
        for page in doc:
            # Create a new page with the same dimensions
            new_page = out_doc.new_page(width=page.rect.width, height=page.rect.height)
            # Retrieve content blocks as dictionary; type == 1 identifies images
            try:
                blocks = page.get_text("dict").get("blocks", [])
            except Exception:
                blocks = []
            for block in blocks:
                if block.get('type') == 1:
                    # Extract bounding box and binary image data
                    bbox = block.get('bbox')
                    image_bytes = block.get('image')
                    if not bbox or not image_bytes:
                        continue
                    rect = fitz.Rect(bbox)
                    # Insert the image into the new page at its original position
                    try:
                        new_page.insert_image(rect, stream=image_bytes)
                    except Exception:
                        # If insertion fails, skip this image
                        pass
        # Save the resulting PDF
        out_doc.save(output_path)
        out_doc.close()
    return output_path


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
    file = request.files['pdf_file']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))
    # Only accept PDF files based on content type
    if not file or not file.filename.lower().endswith('.pdf'):
        flash('Please upload a PDF file')
        return redirect(url_for('index'))

    # Save the uploaded file to a temporary location
    tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    file.save(tmp_in.name)
    input_path = tmp_in.name

    # Determine if PDF contains text; if not, run OCR
    if not pdf_has_text(input_path):
        input_path = run_ocr(input_path)

    # Extract images and create a new PDF
    output_path = extract_images_only(input_path)

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