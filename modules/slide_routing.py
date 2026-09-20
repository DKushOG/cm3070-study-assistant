"""Decide, per slide, whether the vision model is worth calling.

The system previously chose one extraction path for a whole deck. Every page
was either read cheaply from the text layer or sent to the vision-language
model, and the choice was made once. Measurement across 111 corpus slides
showed that to be wrong in both directions at once: 52 per cent of slides
gained at least 30 per cent novel content from the vision call, and the other
48 per cent received an eight second model call that largely re-transcribed
text the PDF text layer already held perfectly.

This module makes the choice per slide, from features computable before any
model call is made.

The rule
--------
On PowerPoint input, call the vision model when pictures occupy at least 10
per cent of the slide. Measured on 81 PowerPoint slides that gives recall
0.85 and precision 0.85 while removing 43 per cent of the model calls.

Picture area is by some distance the best predictor available. Correlated
against the novel content a vision call actually produced, it scores +0.674,
where text yield scores -0.485 and everything else tested scores below 0.3.

On PDF input that feature is not recoverable. A deck built on full bleed
background artwork reports an image area of 1.000 on every page once
rendered, which is exactly what one corpus deck does, and the PDF-derived
version of the same idea collapses from +0.674 to +0.233. The fallback is
text yield: call the vision model when the text layer returns fewer than 60
words. That gives recall 0.74 and precision 0.63, saving 39 per cent. It is
reported as the weaker rule rather than averaged with the stronger one.

Why the thresholds sit where they do
------------------------------------
Recall is preferred over precision at the margin, because the two errors are
not symmetric. A slide wrongly skipped loses its diagram description
permanently and silently. A slide wrongly sent costs about eight seconds.
Raising the picture-area threshold to 0.20 would save 60 per cent of calls
instead of 43, but recall falls from 0.85 to 0.61, and losing a quarter of
the diagram content to save half a minute on a deck is a bad trade for a
revision tool.

This module is imported by evaluation/route_instrumentation.py, which is the
harness the threshold was derived from, so the shipped rule and the measured
rule cannot drift apart.
"""
from pathlib import Path

# Route to the vision model at or above this fraction of the slide given over
# to pictures. PowerPoint input only.
PICTURE_AREA_THRESHOLD = 0.10

# Fallback for PDF input, where picture area is not measurable: route to the
# vision model when the text layer yields fewer than this many words.
TEXT_WORDS_THRESHOLD = 60

ROUTE_VISION = "vision"
ROUTE_TEXT = "text layer"

# Written into the assembled output so a reader, and the evaluation, can see
# which path produced each page. Kept short because it sits in the extracted
# text a student may read.
ROUTE_MARKER = "[extracted by: {route}]"


def pptx_picture_area_ratios(path):
    """Return {slide_number: fraction of the slide occupied by pictures}.

    Read straight from the PowerPoint file, so it costs no rendering and no
    model call. Returns an empty mapping for any input that is not a readable
    pptx, which is what makes the caller fall back to text yield.

    Overlapping pictures are summed and the total is capped at 1.0, so the
    value means "how much of this slide is given over to imagery" rather than
    an exact non-overlapping area.
    """
    if Path(str(path)).suffix.lower() != ".pptx":
        return {}
    try:
        from pptx import Presentation
        from pptx.enum.shapes import MSO_SHAPE_TYPE
    except ImportError:
        return {}
    try:
        presentation = Presentation(str(path))
    except Exception:
        return {}
    slide_area = float(presentation.slide_width) * float(
        presentation.slide_height)
    if not slide_area:
        return {}
    ratios = {}
    for number, slide in enumerate(presentation.slides, start=1):
        covered = 0.0
        for shape in slide.shapes:
            if shape.shape_type != MSO_SHAPE_TYPE.PICTURE:
                continue
            try:
                covered += float(shape.width) * float(shape.height)
            except (TypeError, ValueError):
                continue
        ratios[number] = min(covered / slide_area, 1.0)
    return ratios


def decide(picture_area_ratio=None, text_layer_words=None,
           picture_threshold=None, words_threshold=None):
    """Return (route, reason) for one slide.

    picture_area_ratio is used when it is available, because it is the far
    better predictor. text_layer_words is the fallback. The reason string is
    returned so a run can be audited afterwards rather than only counted.
    """
    if picture_threshold is None:
        picture_threshold = PICTURE_AREA_THRESHOLD
    if words_threshold is None:
        words_threshold = TEXT_WORDS_THRESHOLD

    if picture_area_ratio is not None:
        if picture_area_ratio >= picture_threshold:
            return ROUTE_VISION, (
                f"pictures cover {picture_area_ratio:.0%} of the slide, at or "
                f"above the {picture_threshold:.0%} threshold")
        return ROUTE_TEXT, (
            f"pictures cover only {picture_area_ratio:.0%} of the slide")

    if text_layer_words is not None:
        if text_layer_words < words_threshold:
            return ROUTE_VISION, (
                f"the text layer yields only {text_layer_words} words, below "
                f"the {words_threshold} word threshold")
        return ROUTE_TEXT, (
            f"the text layer yields {text_layer_words} words")

    # Nothing to decide on. Prefer the vision model, because the cost of
    # skipping a slide that needed it is permanent and the cost of calling
    # one that did not is seconds.
    return ROUTE_VISION, "no routing features available, defaulting to vision"


def summarise(report):
    """One plain line describing what a routed run did, for logs and the UI."""
    if not report:
        return ""
    total = report.get("pages", 0)
    vision = report.get("vision_pages", 0)
    if not total:
        return ""
    saved = total - vision
    return (f"{vision} of {total} slides sent to the vision model, "
            f"{saved} read from the text layer "
            f"({(saved / total):.0%} of model calls avoided).")


def extract_routed(input_path, client, dpi=None, max_pages=None,
                   picture_threshold=None, words_threshold=None,
                   on_page=None):
    """Extract a deck, choosing a path per slide instead of per deck.

    A PowerPoint file is converted to PDF once, exactly as the extractor
    comparison does, so every input is handled identically after that point
    and the routing rule does not need to know about PowerPoint. Picture
    areas are read from the original pptx, because that is the feature the
    rule prefers and it is not recoverable from the rendered page.

    Each slide then goes one of two ways. A slide the rule sends to the
    vision model is rendered and extracted with the same retry and empty
    reply detection a whole-deck run uses. A slide it does not send is taken
    from the PDF text layer at no cost at all.

    on_page, when given, is called as on_page(number, route, reason, seconds)
    after each slide, so a long run can report progress without this function
    knowing anything about logging.

    Returns (text, report).
    """
    import time
    import tempfile
    import base64

    from modules import slide_vision, slides_pptx

    try:
        import fitz
    except ImportError as error:
        raise RuntimeError(
            "PyMuPDF is not installed. Install it with 'pip install pymupdf' "
            "to route slides page by page.") from error

    if dpi is None:
        import config
        dpi = config.VISION_DPI

    source = Path(str(input_path))
    picture_ratios = pptx_picture_area_ratios(source)

    workspace = None
    pdf_path = source
    try:
        if source.suffix.lower() == slides_pptx.PPTX_SUFFIX:
            workspace = tempfile.TemporaryDirectory()
            pdf_path = Path(slides_pptx.convert_to_pdf(str(source),
                                                       workspace.name))

        document = fitz.open(str(pdf_path))
        try:
            total = len(document)
            limit = total if not max_pages else min(total, max_pages)

            parts = []
            routes = []
            vision_pages = 0
            unreadable = []
            truncated = []
            retries = 0
            vision_seconds = 0.0

            for index in range(limit):
                number = index + 1
                page = document[index]
                layer_text = page.get_text("text") or ""
                ratio = picture_ratios.get(number)
                route, reason = decide(
                    picture_area_ratio=ratio,
                    text_layer_words=len(layer_text.split()),
                    picture_threshold=picture_threshold,
                    words_threshold=words_threshold)

                started = time.perf_counter()
                if route == ROUTE_VISION:
                    pixmap = page.get_pixmap(dpi=dpi)
                    url = ("data:image/png;base64," + base64.b64encode(
                        pixmap.tobytes("png")).decode("ascii"))
                    text, info = slide_vision.extract_one_image(url, client)
                    vision_pages += 1
                    retries += info["retries"]
                    if info["truncated"]:
                        truncated.append(number)
                    if info["unreadable"]:
                        unreadable.append(number)
                else:
                    text = layer_text.strip()
                elapsed = time.perf_counter() - started
                if route == ROUTE_VISION:
                    vision_seconds += elapsed

                routes.append({"page": number, "route": route,
                               "reason": reason,
                               "picture_area_ratio": ratio,
                               "text_layer_words": len(layer_text.split()),
                               "seconds": round(elapsed, 2)})
                header = (f"{slide_vision.PAGE_MARKER} {number} --- "
                          + ROUTE_MARKER.format(route=route))
                parts.append(f"{header}\n{text}".strip())

                if on_page is not None:
                    on_page(number, route, reason, elapsed)
        finally:
            document.close()
    finally:
        if workspace:
            workspace.cleanup()

    report = {
        "pages": limit,
        "vision_pages": vision_pages,
        "text_pages": limit - vision_pages,
        "unreadable_pages": unreadable,
        "truncated_pages": truncated,
        "retries": retries,
        "vision_seconds": round(vision_seconds, 2),
        "routes": routes,
    }
    return "\n\n".join(parts).strip(), report
