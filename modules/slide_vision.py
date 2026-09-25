"""Extract text and visual information from lecture slides.

This is the project's third pre-trained model and the only one that reads
images. It is an alternative extraction path producing slide text, the same
output as modules/slides_ocr.py, so it fits the existing input modes without
adding a fourth source. What it adds over Tesseract is a description of the
diagrams, figures and charts on a slide as well as the text.

PDF pages are rendered to images with PyMuPDF, chosen over pdf2image because
it installs with pip alone and needs no system package such as poppler, which
keeps the project runnable on Windows. Image files are encoded and sent
directly.

One model call is made per page and the results are joined with a page marker.
The client is passed in, so the module can be tested with a fake one and uses
the same sampling settings as the rest of the system.

If PyMuPDF is missing, is_available() returns False and a call that needs it
raises a RuntimeError with the install command, the same pattern as
modules/audio_stt.py.
"""
import base64
from pathlib import Path

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
DEFAULT_DPI = 150
DEFAULT_MAX_PAGES = 20
PAGE_MARKER = "--- Slide page"

# The instruction sent to the vision model, kept as a constant so the tests
# can check it. It asks for the text word for word and a description of any
# visual content, since describing a diagram is what Tesseract cannot do.
VISION_INSTRUCTION = (
    "You are extracting the content of a single lecture slide image.\n"
    "Return exactly two sections, both always present, in this order.\n\n"
    "TEXT:\n"
    "Transcribe all visible text on the slide verbatim, preserving reading "
    "order.\n\n"
    "VISUALS:\n"
    "Describe every diagram, figure, chart, photograph or illustration on "
    "the slide, including what it shows, its labels, and the relationships "
    "it depicts. If the slide genuinely contains no visual element beyond "
    "text, write exactly: none.\n\n"
    "Do not add facts that are not present on the slide."
)

# An earlier version of the instruction ended "If the slide has no diagram,
# transcribe the text only." The model took that escape. On a slide with three
# labelled figures it returned the text alone and no description, which is the
# one thing a vision model is here for. Requiring both sections, with an
# explicit "none" for the empty case, fixed that page and changed nothing on
# slides that really have no visual content.
#
# A JSON schema at decode time, the technique used for the quiz, was tried
# here and dropped. It did not improve on the wording above and one call ended
# in a server error. The two problems are different shapes. The quiz left out a
# required line, which a prompt has no way to stop, while this model was taking
# an escape the prompt itself offered. Removing the escape was enough, and was
# roughly two and a half times faster per page.

# Written in place of a slide the model could not read, so a missing page is
# visible to the reader instead of just absent.
UNREADABLE_NOTE = "[This slide could not be read by the vision model.]"


def is_available():
    """Return whether PDF page rendering is possible, meaning PyMuPDF is
    installed.

    This reports local rendering only. Whether the vision model answers is
    something the client reports when a call is made, so True here does not
    mean an extraction will succeed.
    """
    try:
        import fitz  # noqa: F401  (PyMuPDF)
        return True
    except ImportError:
        return False


def encode_image_file(path):
    """Read an image file and return it as a base64 data URL.

    Uses the standard library only, so sending a single image needs no
    optional package. Only PDF rendering does.
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


def _call_one_page(client, url, max_tokens):
    """Call the model for one page, returning (text, truncated).

    Asks for the reply metadata where the client supports it and falls back to
    a plain call where it does not, so a stand-in client with the older
    two-argument signature still works.
    """
    try:
        result = client.chat_with_images(VISION_INSTRUCTION, [url],
                                         max_tokens=max_tokens,
                                         with_meta=True)
    except TypeError:
        return (client.chat_with_images(VISION_INSTRUCTION, [url]) or ""), False
    if isinstance(result, tuple):
        text, meta = result
        return (text or ""), (meta or {}).get("finish_reason") == "length"
    return (result or ""), False


def extract_one_image(url, client, max_tokens=None, max_retries=None):
    """Extract one slide image, retrying if the reply comes back unusable.

    Returns (text, info), where info says whether the reply was cut short,
    whether the page ended up unreadable and how many retries were used.

    Public because per-slide routing extracts one page at a time and uses this
    function, so both paths share the same retry and detection code.
    """
    import config

    if max_tokens is None:
        max_tokens = config.VISION_NUM_PREDICT or None
    if max_retries is None:
        max_retries = config.VISION_MAX_EMPTY_RETRIES

    text, truncated = _call_one_page(client, url, max_tokens)
    retries = 0
    # Worth another try when the page returned nothing, or stopped early and
    # is incomplete. Retrying a page that answered in full only costs time.
    while (not text.strip() or truncated) and retries < max_retries:
        retries += 1
        text, truncated = _call_one_page(client, url, max_tokens)

    unreadable = not text.strip()
    if unreadable:
        text = UNREADABLE_NOTE
    return text, {"truncated": truncated, "unreadable": unreadable,
                  "retries": retries}


def extract_from_images_with_report(images, client, max_tokens=None,
                                    max_retries=None):
    """Extract slide text from rendered images, reporting what failed.

    One model call per image, joined with a page marker. A page that comes back
    empty, or stops because it ran out of room, is retried and then recorded
    instead of being passed on as a success.

    The earlier version reported 49 of 111 slides as extracted when the model
    had returned an empty string for each of them. An empty reply and a blank
    slide look the same to a string join, so the fault did not show up in any
    measurement taken over it, including one that reached the draft report.

    Returns (text, report), where report holds the page count, the pages that
    could not be read, the pages that were cut short and how many retries were
    used. Callers that do not need the report use extract_from_images.
    """
    import config

    if max_tokens is None:
        max_tokens = config.VISION_NUM_PREDICT or None
    if max_retries is None:
        max_retries = config.VISION_MAX_EMPTY_RETRIES

    parts = []
    unreadable = []
    truncated = []
    retries = 0

    for number, url in enumerate(images, start=1):
        text, info = extract_one_image(url, client, max_tokens=max_tokens,
                                       max_retries=max_retries)
        retries += info["retries"]
        if info["truncated"]:
            truncated.append(number)
        if info["unreadable"]:
            unreadable.append(number)
        parts.append(f"{PAGE_MARKER} {number} ---\n{text}".strip())

    report = {
        "pages": len(images),
        "unreadable_pages": unreadable,
        "truncated_pages": truncated,
        "retries": retries,
    }
    return "\n\n".join(parts).strip(), report


def describe_report(report):
    """One plain line summarising a report, for the interface and the logs.

    Returns an empty string when every page was read, so a caller can print it
    without checking first.
    """
    if not report:
        return ""
    unreadable = report.get("unreadable_pages") or []
    truncated = report.get("truncated_pages") or []
    if not unreadable and not truncated:
        return ""
    pieces = []
    if unreadable:
        pieces.append(
            f"{len(unreadable)} of {report.get('pages', 0)} slides could not "
            f"be read (pages {', '.join(str(p) for p in unreadable)})")
    if truncated:
        pieces.append(
            f"{len(truncated)} were cut short "
            f"(pages {', '.join(str(p) for p in truncated)})")
    return ". ".join(pieces) + "."


def extract_from_images(images, client):
    """Extract slide text from already rendered images.

    One model call per image, joined with a page marker. Kept separate from
    extract_slides so the comparison harness can render the pages once and give
    the same images to the Tesseract path.

    Keeps its original signature, so existing callers are unaffected. Use
    extract_from_images_with_report to find out which pages failed.
    """
    text, _report = extract_from_images_with_report(images, client)
    return text


def extract_slides(path, client, dpi=DEFAULT_DPI, max_pages=DEFAULT_MAX_PAGES):
    """Extract slide text from a PDF or image file using the vision model.

    Checks the file type before any optional import, so an unsupported type
    fails straight away with a clear message, as modules/slides_ocr.py does.
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
