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

# The instruction above replaced an earlier version ending "If the slide has
# no diagram, transcribe the text only." That closing sentence was an escape
# clause and the model took it: on a slide carrying three labelled figures it
# returned transcription alone and no description, which is precisely the
# capability that justifies a vision model over Tesseract. Requiring both
# sections to exist, with an explicit "none" for the empty case, fixed that
# page and left correct behaviour unchanged on slides that genuinely have no
# visual content.
#
# A JSON schema enforced at decode time, the technique used for quiz
# generation, was also tested here and rejected. It gave no improvement over
# the wording above and aborted one call with a server error. The two faults
# are not the same shape: the quiz omitted a structurally required line,
# which a prompt cannot prevent, whereas this model was taking a
# discretionary escape the prompt itself offered. Removing the escape was
# sufficient, and was also about two and a half times faster per page.

# Written in place of a slide the model could not read, so a missing page is
# visible to a reader rather than silently absent.
UNREADABLE_NOTE = "[This slide could not be read by the vision model.]"


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


def _call_one_page(client, url, max_tokens):
    """One page, returning (text, truncated).

    Asks for metadata where the client supports it, and falls back to a plain
    call for any client that does not, so a test double implementing only the
    original two-argument signature still works.
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
    """Extract one slide image, retrying once if it comes back unusable.

    Returns (text, info) where info carries whether the reply was truncated,
    whether the page ended up unreadable, and how many retries were spent.
    Public because the per-slide routing path extracts one page at a time and
    must get the same retry and detection behaviour as a whole-deck run,
    rather than a second implementation of it that could drift.
    """
    import config

    if max_tokens is None:
        max_tokens = config.VISION_NUM_PREDICT or None
    if max_retries is None:
        max_retries = config.VISION_MAX_EMPTY_RETRIES

    text, truncated = _call_one_page(client, url, max_tokens)
    retries = 0
    # A page is worth one more try when it produced nothing at all, or when
    # it stopped early and so is incomplete. Retrying a page that answered
    # fully would only cost time.
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

    One model call per image, joined with a page marker, exactly as before.
    What is new is that a page which comes back empty, or which stops because
    it ran out of room, is retried once and then recorded rather than passed
    through as if it had succeeded.

    This exists because the silent version of this function reported 49 of
    111 slides as extracted when the model had returned an empty string for
    each of them. An empty reply and a blank slide are the same thing to a
    string join, so the fault was invisible to every measurement taken over
    it, including one that reached the draft report.

    Returns (text, report) where report is a dictionary carrying the page
    count, the page numbers that could not be read, the numbers that were
    truncated, and how many retries were spent. Callers that do not want the
    report use extract_from_images, which is unchanged.
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
    """A one-line plain summary of a report, for the interface and the logs.

    Returns an empty string when every page was read, so a caller can print
    it unconditionally and say nothing when there is nothing to say.
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
    """Extract slide text from already-rendered images using the client.

    One model call per image, joined with a page marker. Kept separate from
    extract_slides so the extractor comparison harness can render pages once
    and share the identical images with the Tesseract path.

    Retains its original signature and return type so every existing caller
    is unaffected. Callers that need to know which pages failed should use
    extract_from_images_with_report instead.
    """
    text, _report = extract_from_images_with_report(images, client)
    return text


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
