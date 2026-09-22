"""Choose and run a slide extraction method for an uploaded file.

The interface offers every extraction method that can run for the file the
student uploads. This module holds that logic so it can be tested without
Streamlit, and so the interface runs the same code paths as the evaluation
harness.

Four methods exist. Native PowerPoint text keeps slide structure and needs
nothing beyond python-pptx. The classical path reads a PDF text layer or runs
Tesseract. The vision model reads every page. Per-slide routing, from
modules/slide_routing.py, reads each slide from its text layer unless the
routing rule says the vision model will add something, which is the
configuration the Evaluation chapter measured and the one offered first.

Both paths that call the vision model use the reply detection in
modules/slide_vision.py, so a slide the model could not read is marked in
the text and reported to the caller, never passed through as an empty page.
"""
from pathlib import Path

import config
from modules import slide_routing, slide_vision, slides_ocr, slides_pptx

NATIVE_PPTX_PATH = "Native PowerPoint text (python-pptx)"
CLASSICAL_PATH = "Classical (pypdf / Tesseract OCR)"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def vision_path_label():
    """Label for the whole-deck vision method, naming the configured model."""
    return f"Vision model ({config.VISION_MODEL})"


def routed_path_label():
    """Label for per-slide routing, naming the configured vision model."""
    return f"Per-slide routing (text layer, or {config.VISION_MODEL} where needed)"


def paths_for(suffix):
    """Return the extraction methods that can run for this file type.

    A pptx reaches the pixel-based methods only through a LibreOffice
    conversion, so those are offered for a deck only when LibreOffice is
    present. The native text method never needs it. Routing needs a whole
    document to route across, so it is not offered for a single image.

    Per-slide routing, when available, is listed first because it is the
    default the system is designed around.
    """
    suffix = (suffix or "").lower()
    is_pptx = suffix == slides_pptx.PPTX_SUFFIX
    is_image = suffix in IMAGE_SUFFIXES
    pdf_reachable = suffix == ".pdf" or (
        is_pptx and slides_pptx.libreoffice_available())
    paths = []
    if pdf_reachable and slide_vision.is_available():
        paths.append(routed_path_label())
    if is_pptx and slides_pptx.is_available():
        paths.append(NATIVE_PPTX_PATH)
    if ((pdf_reachable and (slides_ocr.pdf_available()
                            or slides_ocr.ocr_available()))
            or (is_image and slides_ocr.ocr_available())):
        paths.append(CLASSICAL_PATH)
    if (pdf_reachable or is_image) and slide_vision.is_available():
        paths.append(vision_path_label())
    return paths


def needs_vision_model(method):
    """Return whether this method calls the vision model."""
    return method in (vision_path_label(), routed_path_label())


def extract(path, method, vision_client=None, max_pages=None):
    """Extract slide text from the file at path using the chosen method.

    Returns (text, notice). notice is a plain sentence for the interface
    describing what routing did or which slides could not be read, and is
    empty when there is nothing to report.

    max_pages defaults to config.VISION_MAX_PAGES and is applied to every
    method that renders pages, so the whole deck is read unless the cap is
    deliberately lowered.
    """
    if max_pages is None:
        max_pages = config.VISION_MAX_PAGES
    suffix = Path(str(path)).suffix.lower()
    is_pptx = suffix == slides_pptx.PPTX_SUFFIX

    if method == NATIVE_PPTX_PATH:
        # Speaker notes stay off: they are content no other method can see,
        # so including them would confound the extraction comparison.
        return slides_pptx.extract_pptx_text(path, include_notes=False), ""

    if method == routed_path_label():
        text, report = slide_routing.extract_routed(
            path, vision_client, dpi=config.VISION_DPI, max_pages=max_pages,
            picture_threshold=config.SLIDE_ROUTING_PICTURE_THRESHOLD,
            words_threshold=config.SLIDE_ROUTING_WORDS_THRESHOLD)
        notice = " ".join(part for part in (
            slide_routing.summarise(report),
            slide_vision.describe_report(report)) if part)
        return text, notice

    if method == vision_path_label():
        if is_pptx:
            images = slides_pptx.render_pptx_to_images(
                path, dpi=config.VISION_DPI, max_pages=max_pages)
        elif suffix == ".pdf":
            images = slide_vision.render_pdf_to_images(
                path, dpi=config.VISION_DPI, max_pages=max_pages)
        elif suffix in IMAGE_SUFFIXES:
            images = [slide_vision.encode_image_file(path)]
        else:
            raise ValueError(
                f"Unsupported slide file type '{suffix}'. "
                "Provide a .pptx, .pdf, .png or .jpg file.")
        text, report = slide_vision.extract_from_images_with_report(
            images, vision_client)
        return text, slide_vision.describe_report(report)

    if method == CLASSICAL_PATH:
        if is_pptx:
            # Convert once, then reuse the existing PDF path so nothing
            # downstream needs to know about PowerPoint.
            import tempfile
            with tempfile.TemporaryDirectory() as workspace:
                pdf_path = slides_pptx.convert_to_pdf(str(path), workspace)
                return slides_ocr.extract_slides(pdf_path), ""
        return slides_ocr.extract_slides(path), ""

    raise ValueError(f"Unknown slide extraction method '{method}'.")
