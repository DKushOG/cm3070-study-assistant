"""Slide and PDF text extraction module (Phase B extension point).

Two paths: digital PDFs use the embedded text layer through pypdf, and
slide images use Tesseract OCR through pytesseract. Both degrade
gracefully so the app runs without them and the user can paste slide text
manually.
"""
from pathlib import Path

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def pdf_available():
    try:
        import pypdf  # noqa: F401
        return True
    except ImportError:
        return False


def ocr_available():
    try:
        import pytesseract  # noqa: F401
        from PIL import Image  # noqa: F401
        return True
    except ImportError:
        return False


def extract_pdf_text(path):
    try:
        from pypdf import PdfReader
    except ImportError as error:
        raise RuntimeError(
            "pypdf is not installed. Install it with 'pip install pypdf'."
        ) from error
    reader = PdfReader(str(path))
    pages = [(page.extract_text() or "") for page in reader.pages]
    return "\n\n".join(part for part in pages if part.strip()).strip()


def ocr_image(path):
    try:
        import pytesseract
        from PIL import Image
    except ImportError as error:
        raise RuntimeError(
            "pytesseract and Pillow are required for OCR. Install them with "
            "'pip install pytesseract Pillow' and install the Tesseract "
            "engine on the system."
        ) from error
    with Image.open(str(path)) as image:
        return pytesseract.image_to_string(image).strip()


def extract_slides(path):
    """Dispatch extraction by file type. Suffix checks happen before any
    optional import so unsupported types fail fast with a clear message."""
    suffix = Path(str(path)).suffix.lower()
    if suffix == ".pdf":
        return extract_pdf_text(path)
    if suffix in IMAGE_SUFFIXES:
        return ocr_image(path)
    raise ValueError(
        f"Unsupported slide file type '{suffix}'. "
        "Provide a .pdf, .png or .jpg file."
    )
