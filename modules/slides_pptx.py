"""PowerPoint slide extraction module.

pptx is the dominant slide format, so refusing it contradicted the promise in
Table 3.1 of the Design chapter to accept mixed lecture materials and the
Chapter 1.3 commitment to common learning material formats. A student should
be able to feed in the deck they already have, unmodified; requiring a manual
PDF export first would undermine the premise of the project. This module
removes that friction.

Two deliberately separate capabilities, because one is cheap and one is not:

1. Text extraction through python-pptx: pure pip, no system dependency, and it
   reads the slide's own structure. python-pptx is a parser, in the same
   category as pypdf - it reads the file format directly and does no
   inference. Its advantage over OCR is that slide boundaries, titles and
   table cells survive, where OCR flattens a rendered page into loose lines.

2. Rendering to pixels for the OCR and vision paths, which needs LibreOffice,
   because nothing in pure Python renders PowerPoint faithfully. This is
   optional: when LibreOffice is absent the text path still works and the
   interface says what is missing, so a missing heavy dependency degrades to a
   clear message rather than a crash, following modules/audio_stt.py.

Speaker notes are extracted only when explicitly requested and are off by
default. This is a fairness requirement, not a feature preference: speaker
notes are content no other extraction path can see, so including them by
default would confound the four-way comparison in section 5.2 - the native
path would appear to win purely by having access to material the OCR and
vision paths never receive. It is the same argument as rendering each page
once and sharing the images in evaluation/compare_extractors.py.
"""
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from modules.slide_vision import PAGE_MARKER

PPTX_SUFFIX = ".pptx"

# The default Windows install location, checked as well as PATH because the
# LibreOffice installer does not add itself to PATH on Windows.
WINDOWS_SOFFICE = r"C:\Program Files\LibreOffice\program\soffice.exe"

# Seconds to allow for a headless conversion before giving up, so a stalled
# LibreOffice cannot hang the interface indefinitely.
CONVERT_TIMEOUT = 180

NOTES_MARKER = "[Speaker notes]"


def is_available():
    """Return whether pptx text extraction is possible (python-pptx installed).

    Text extraction is the capability this reports. Rendering for the OCR and
    vision paths is reported separately by libreoffice_available, because the
    two have different dependencies and only one of them is pure pip.
    """
    try:
        import pptx  # noqa: F401  (python-pptx)
        return True
    except ImportError:
        return False


def find_soffice():
    """Locate the LibreOffice binary, or return None.

    Checks PATH first, then the default Windows install location, because the
    Windows installer does not put soffice.exe on PATH.
    """
    found = shutil.which("soffice")
    if found:
        return found
    if os.path.isfile(WINDOWS_SOFFICE):
        return WINDOWS_SOFFICE
    return None


def libreoffice_available():
    """Return whether pptx can be rendered to pixels (LibreOffice present)."""
    return find_soffice() is not None


def _shape_text(shape):
    """Return the text of one shape, including table cells.

    Tables are read cell by cell and joined with tabs so the row structure
    survives in the extracted text, which is the structural advantage this
    path has over OCR.
    """
    parts = []
    if getattr(shape, "has_table", False):
        for row in shape.table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            if any(cells):
                parts.append("\t".join(cells))
    elif getattr(shape, "has_text_frame", False):
        text = shape.text_frame.text.strip()
        if text:
            parts.append(text)
    return parts


def _slide_title(slide):
    try:
        placeholder = slide.shapes.title
    except (AttributeError, ValueError):
        return ""
    if placeholder is None:
        return ""
    return (placeholder.text or "").strip()


def _notes_text(slide):
    if not getattr(slide, "has_notes_slide", False):
        return ""
    frame = getattr(slide.notes_slide, "notes_text_frame", None)
    if frame is None:
        return ""
    return (frame.text or "").strip()


def extract_pptx_text(path, include_notes=False):
    """Extract slide text from a pptx file, preserving slide boundaries.

    Uses the same page marker as modules/slide_vision.py so outputs from
    different extraction paths line up directly in the section 5.2 comparison
    table. The slide title is emitted first when present, then the remaining
    shape and table text in document order.

    include_notes is off by default on purpose: see the module docstring. When
    switched on, notes are appended under a clearly labelled marker so they are
    never mistaken for text that was visible on the slide.

    Raises RuntimeError with an installation hint if python-pptx is missing.
    """
    try:
        from pptx import Presentation
    except ImportError as error:
        raise RuntimeError(
            "python-pptx is not installed. Install it with "
            "'pip install python-pptx' to read PowerPoint files."
        ) from error
    presentation = Presentation(str(path))
    pages = []
    for number, slide in enumerate(presentation.slides, start=1):
        lines = []
        title = _slide_title(slide)
        if title:
            lines.append(title)
        for shape in slide.shapes:
            for text in _shape_text(shape):
                # The title placeholder is also a shape; do not repeat it.
                if text != title:
                    lines.append(text)
        if include_notes:
            notes = _notes_text(slide)
            if notes:
                lines.append(f"{NOTES_MARKER}\n{notes}")
        body = "\n".join(lines).strip()
        pages.append(f"{PAGE_MARKER} {number} ---\n{body}".strip())
    return "\n\n".join(pages).strip()


def convert_to_pdf(path, output_dir):
    """Convert a pptx to PDF headlessly with LibreOffice and return the path.

    Converting to PDF means the existing PDF rendering path is reused
    unchanged, so the OCR and vision paths need no knowledge of PowerPoint at
    all. Raises RuntimeError with a clear message when LibreOffice is absent or
    the conversion fails, which is what lets the interface explain the missing
    dependency instead of crashing.
    """
    soffice = find_soffice()
    if soffice is None:
        raise RuntimeError(
            "LibreOffice was not found, so this PowerPoint file cannot be "
            "rendered to images for the OCR or vision paths. Either install "
            "LibreOffice, or export the deck to PDF from PowerPoint and "
            "upload that PDF instead. Native pptx text extraction still works "
            "without LibreOffice."
        )
    try:
        completed = subprocess.run(
            [soffice, "--headless", "--convert-to", "pdf", "--outdir",
             str(output_dir), str(path)],
            capture_output=True, text=True, timeout=CONVERT_TIMEOUT,
            check=False)
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(
            f"LibreOffice conversion failed to start: {error}") from error
    produced = Path(output_dir) / (Path(str(path)).stem + ".pdf")
    if not produced.is_file():
        raise RuntimeError(
            "LibreOffice did not produce a PDF for this file "
            f"(exit code {completed.returncode}). "
            "Export the deck to PDF from PowerPoint and upload that instead."
        )
    return str(produced)


def render_pptx_to_images(path, dpi=None, max_pages=None):
    """Render pptx slides to images by converting to PDF first.

    The temporary PDF is discarded once the pages are rendered; only the images
    are returned, so the caller is unaware a conversion happened.
    """
    from modules import slide_vision
    kwargs = {}
    if dpi is not None:
        kwargs["dpi"] = dpi
    if max_pages is not None:
        kwargs["max_pages"] = max_pages
    with tempfile.TemporaryDirectory() as workspace:
        pdf_path = convert_to_pdf(path, workspace)
        return slide_vision.render_pdf_to_images(pdf_path, **kwargs)
