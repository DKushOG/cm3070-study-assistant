"""Read slide text without a model, using the classical tools.

Two paths. A digital PDF is read through pypdf, which returns the text layer
the file already carries. A slide image is read with Tesseract through
pytesseract.

Either package can be missing. The availability checks let the app hide the
method instead of failing, and the user can paste slide text by hand.
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
    """Choose the path by file type.

    The suffix is checked before any optional import, so an unsupported type
    fails straight away with a clear message.
    """
    suffix = Path(str(path)).suffix.lower()
    if suffix == ".pdf":
        return extract_pdf_text(path)
    if suffix in IMAGE_SUFFIXES:
        return ocr_image(path)
    raise ValueError(
        f"Unsupported slide file type '{suffix}'. "
        "Provide a .pdf, .png or .jpg file."
    )
