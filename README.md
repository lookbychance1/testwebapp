# SharePremium PDF Image Extractor

This repository contains a simple Flask‑based web application that accepts a
PDF upload, identifies all images on each page, removes any other content
(such as text, vector graphics or watermarks) and returns a new PDF
containing only the images.  If the uploaded file is a scanned document
without any selectable text, the application will automatically run
OCR using [OCRmyPDF](https://ocrmypdf.readthedocs.io/en/latest/introduction.html)
to make it searchable before extracting images.【422720727765131†L46-L53】  Images are
identified via PyMuPDF's `get_text("dict")` method, which returns a list
of content blocks; blocks of type `1` represent image objects【740116005133473†L274-L297】.

## Features

* Upload a PDF through a web form.
* Automatically determine if the PDF contains any selectable text.
* Perform OCR on scanned PDFs using **OCRmyPDF** (if installed).
* Create a new PDF where only images are retained; all text, watermarks and
  other vector graphics are removed.
* Provide a one‑click download link for the processed PDF.

## Installation

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Make sure that **Tesseract OCR** and **OCRmyPDF** are installed on the system
if you want the OCR step to run.  Without OCRmyPDF the application still
works, but scanned documents will not be converted to searchable PDFs.

## Running Locally

You can start the development server with:

```bash
python app.py
```

Then open [http://localhost:5000](http://localhost:5000) in your browser and
upload a PDF to test.

For deployment on Render, use the provided `Procfile` and set the build
command to `pip install -r requirements.txt` and the start command to
`gunicorn --bind 0.0.0.0:$PORT app:app`.

## How it works

1. **Uploading** – The user selects a PDF file via the web form.  The
   server saves the file to a temporary location.
2. **Detection** – PyMuPDF opens the PDF and checks if any page contains
   selectable text.  If none is found, the server calls OCRmyPDF to add
   an OCR layer so that text can be detected later【422720727765131†L46-L53】.
3. **Extraction** – The application iterates through each page.  It calls
   `get_text("dict")` to retrieve the page's content blocks and filters
   out those with `type == 1`, which represent images【740116005133473†L274-L297】.  For every
   image block, the binary image data and its bounding box are used to
   insert the image into a new blank page in the output PDF.  This
   preserves the original positions and dimensions of the images while
   discarding all other content.
4. **Download** – The processed file is stored in a temporary location
   and the user is shown a page with a button to download the final
   modified PDF.

## Caveats

* Some scanned PDFs embed the entire page as an image; in these cases
  the whole page will be preserved because it is treated as a single
  image.  Removing text embedded inside such an image is beyond the
  scope of this simple script.
* The in‑memory dictionary used to store processed files is not
  persistent.  If the server restarts, the references will be lost
  and existing download links will no longer work.
