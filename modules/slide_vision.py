"""Vision-based slide extraction module (third pre-trained model).

This is the project's third pre-trained model operating on a new data space:
a vision-language model reading slide *images*. It is deliberately an
alternative extraction path that produces slide_text, exactly like
modules/slides_ocr.py, so it slots into the four canonical input modes
without adding a fourth source. The point the Evaluation chapter tests is
that a VLM can transcribe slide text *and* describe diagrams, figures and
charts, which classical OCR (Tesseract) cannot.

PDF pages are rendered to images with PyMuPDF (pip install pymupdf), chosen
over pdf2image because it is pure pip with no system dependency such as
poppler, keeping the "always runnable" promise realistic on Windows. Image
files are encoded directly. Extraction runs one model call per page and
joins the results with a clear page marker, and the client is injected so the
module is testable with a fake and honours the shared reproducibility
settings.

Graceful degradation follows the pattern in modules/audio_stt.py: an
is_available() check and a RuntimeError with an install hint when a missing
dependency is actually needed.
"""
import base64
from pathlib import Path

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
DEFAULT_DPI = 150
DEFAULT_MAX_PAGES = 20
PAGE_MARKER = "--- Slide page"

# The instruction sent to the vision model, kept as a module-level constant
# so it is quotable in the report and assertable in tests. It asks for
# verbatim text *and* a description of any visual content, because reading
# diagrams is the capability OCR lacks and the exact thing the evaluation
# measures.
VISION_INSTRUCTION = (
    "You are extracting the content of a single lecture slide image.\n"
    "1. Transcribe all visible text on the slide verbatim, preserving its "
    "reading order.\n"
    "2. Then describe any diagrams, figures, charts or images on the slide, "
    "including what they show and any labels or relationships.\n"
    "Do not add facts that are not present on the slide. If the slide has no "
    "diagram, transcribe the text only."
)


def is_available():
    """Return whether PDF page rendering is possible (PyMuPDF installed).

    This reports the local rendering capability only. Whether the vision
    model itself is reachable is a runtime concern surfaced by the client
    when a call is made, not something this module can check, so a True here
    does not guarantee a successful extraction.
    """
    try:
        import fitz  # noqa: F401  (PyMuPDF)
        return True
    except ImportError:
        return False


def encode_image_file(path):
    """Read an image file and return it as a base64 data URL.

    Uses only the standard library, so encoding an image and calling the
    vision model needs no optional package; only PDF rendering does.
    """
    suffix = Path(str(path)).suffix.lower()
    mime = "jpeg" if suffix in (".jpg", ".jpeg") else "png"
    data = Path(str(path)).read_bytes()
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:image/{mime};base64,{encoded}"


def render_pdf_to_images(path, dpi=DEFAULT_DPI, max_pages=DEFAULT_MAX_PAGES):
    """Render up to max_pages PDF pages to PNG base64 data URLs at the given
    DPI. Raises RuntimeError with an install hint if PyMuPDF is missing."""
    try:
        import fitz
    except ImportError as error:
        raise RuntimeError(
            "PyMuPDF is not installed. Install it with 'pip install pymupdf' "
            "to render PDF slides for the vision model."
        ) from error
    images = []
    with fitz.open(str(path)) as document:
        for index, page in enumerate(document):
            if index >= max_pages:
                break
            pixmap = page.get_pixmap(dpi=dpi)
            encoded = base64.b64encode(pixmap.tobytes("png")).decode("ascii")
            images.append(f"data:image/png;base64,{encoded}")
    return images


def extract_from_images(images, client):
    """Extract slide text from already-rendered images using the client.

    One model call per image, joined with a page marker. Kept separate from
    extract_slides so the extractor comparison harness can render pages once
    and share the identical images with the Tesseract path.
    """
    parts = []
    for number, url in enumerate(images, start=1):
        reply = client.chat_with_images(VISION_INSTRUCTION, [url])
        parts.append(f"{PAGE_MARKER} {number} ---\n{reply}".strip())
    return "\n\n".join(parts).strip()


def extract_slides(path, client, dpi=DEFAULT_DPI, max_pages=DEFAULT_MAX_PAGES):
    """Extract slide text from a PDF or image file using the vision model.

    Dispatches by file type before any optional import so unsupported types
    fail fast with a clear message, mirroring modules/slides_ocr.py.
    """
    suffix = Path(str(path)).suffix.lower()
    if suffix == ".pdf":
        images = render_pdf_to_images(path, dpi=dpi, max_pages=max_pages)
    elif suffix in IMAGE_SUFFIXES:
        images = [encode_image_file(path)]
    else:
        raise ValueError(
            f"Unsupported slide file type '{suffix}'. "
            "Provide a .pdf, .png or .jpg file."
        )
    return extract_from_images(images, client)
