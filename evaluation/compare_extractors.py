"""Extractor comparison harness (component evaluation for section 5.2).

Runs every available slide-extraction path over the same input and records
what each produced, which is the "evidence of testing and rejecting models"
the project brief demands and the justification for the vision module. The
four paths are:

  * Native pptx - python-pptx reading the deck's own structure (a parser, not
                  a model, in the same category as pypdf)
  * pypdf       - the PDF's embedded text layer (no model)
  * Tesseract   - classical OCR on the rendered page images
  * Vision      - the vision-language model on the same rendered page images

A .pptx input is converted to PDF once with LibreOffice, after which the pypdf,
Tesseract and vision paths run exactly as they do for a PDF, so one deck yields
four comparable techniques. Without LibreOffice the native path still runs and
the others are skipped with a message. Speaker notes are never included here:
they are content the OCR and vision paths cannot see, so including them would
make the native path win by having more material rather than by extracting
better, which is the same fairness argument as sharing the rendered pages.

Fairness argument (stated here because it is quoted in the report): Tesseract
cannot read a PDF directly, so each PDF page is rendered to an image *once*
with PyMuPDF and those identical images are shared between the Tesseract and
vision paths. All image-based paths therefore see exactly the same pixels,
and any quality difference is the model's, not the input's.

Unavailable paths are skipped with a warning instead of crashing, mirroring
how evaluation/run_eval.py handles missing sources. The client is injectable
for testing, and the scoring columns are left blank for manual marking, as
in run_eval.py.

Usage:
    python -m evaluation.compare_extractors --input samples/slides.pdf --out outputs
"""
import argparse
import base64
import csv
import os
import tempfile
import time
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path

import config
from core.llm_client import LLMClient
from modules import slide_vision, slides_ocr, slides_pptx

NATIVE_PPTX_PATH = "Native pptx (python-pptx)"
PYPDF_PATH = "pypdf (embedded text layer)"
TESSERACT_PATH = "Tesseract OCR"
VISION_PATH = "Vision model"

# The manual columns must match the OCR 5.2 sheet of the evaluation workbook
# exactly, in this order and spelling; the four score columns and Notes stay
# blank for hand marking. The automatic columns are appended after them.
MANUAL_COLUMNS = [
    "Slide set",
    "Pages",
    "Extraction path",
    "Missing text (1 to 5)",
    "Incorrect text (1 to 5)",
    "Structure kept (1 to 5)",
    "Overall (1 to 5)",
    "Notes",
]
AUTO_COLUMNS = [
    "Character count",
    "Word count",
    "Seconds elapsed",
    "Model used",
    # Blank for every path but the vision one, which is the only path that can
    # fail per page without raising. Recorded next to the word count because
    # a word count computed over an output with empty pages in it is not
    # comparable to one computed over a complete extraction, and reading the
    # two columns together is the only way to notice.
    "Unreadable pages",
    "Truncated pages",
    "Retries spent",
    "Output text file",
]
CSV_COLUMNS = MANUAL_COLUMNS + AUTO_COLUMNS

IMAGE_SUFFIXES = slide_vision.IMAGE_SUFFIXES


def build_paths(suffix, pypdf_ok, ocr_ok, render_ok, pptx_ok=False,
                libreoffice_ok=False):
    """Decide which extraction paths can run for this input.

    Returns (runnable, skipped) where runnable is a list of path names and
    skipped is a list of (name, reason) pairs. A path is runnable only if the
    dependencies it needs are present: the image-based paths need page
    rendering (PyMuPDF) for a PDF, an image file needs no rendering, and a
    pptx needs LibreOffice before it can become pixels at all.
    """
    is_pdf = suffix == ".pdf"
    is_image = suffix in IMAGE_SUFFIXES
    is_pptx = suffix == slides_pptx.PPTX_SUFFIX
    # A pptx reaches the PDF-based paths only once LibreOffice has converted
    # it, so those paths depend on LibreOffice for this input type.
    pdf_reachable = is_pdf or (is_pptx and libreoffice_ok)
    runnable = []
    skipped = []

    def consider(name, ok, reason):
        (runnable if ok else skipped).append(name if ok else (name, reason))

    # The native path reads the deck's own structure; pptx only.
    consider(NATIVE_PPTX_PATH, is_pptx and pptx_ok,
             "needs a .pptx file and python-pptx installed")
    # pypdf reads a PDF's text layer, including one converted from a pptx.
    consider(PYPDF_PATH, pdf_reachable and pypdf_ok,
             "needs a PDF (or LibreOffice to convert a pptx) and pypdf")
    # Tesseract needs OCR installed, plus rendering unless given an image.
    tesseract_ok = ocr_ok and (is_image or (pdf_reachable and render_ok))
    consider(TESSERACT_PATH, tesseract_ok,
             "needs pytesseract, PyMuPDF to render, and LibreOffice for a pptx")
    # The vision model needs rendering; an image is sent directly.
    vision_ok = is_image or (pdf_reachable and render_ok)
    consider(VISION_PATH, vision_ok,
             "needs PyMuPDF to render, and LibreOffice for a pptx")
    return runnable, skipped


def _data_url_to_bytes(url):
    _header, _comma, payload = url.partition(",")
    return base64.b64decode(payload)


def _ocr_shared_images(images):
    """OCR the shared rendered pages by writing each to a temp file and
    reusing slides_ocr.ocr_image, so the OCR path sees the identical pixels
    the vision path sees."""
    parts = []
    for number, url in enumerate(images, start=1):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(_data_url_to_bytes(url))
            tmp_path = tmp.name
        try:
            text = slides_ocr.ocr_image(tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        parts.append(f"{slide_vision.PAGE_MARKER} {number} ---\n{text}".strip())
    return "\n\n".join(parts).strip()


def _run_path(name, input_path, pdf_path, suffix, images, client,
              report_out=None):
    """Run one extraction path and return (text, model_used).

    pdf_path is the original file for a PDF input and the LibreOffice
    conversion for a pptx, so the PDF text layer path is identical either way.

    report_out, when a dictionary is supplied, is filled with the vision
    path's per-page report: which pages could not be read, which were cut
    short, and how many retries were spent. It is optional so the signature
    stays compatible, but it should always be passed here. Without it this
    harness cannot tell an empty reply from a blank slide, and that is
    precisely how the comparison in the draft report came to be computed over
    an output in which 15 of 36 pages were empty.
    """
    if name == NATIVE_PPTX_PATH:
        # Speaker notes stay off: see the module docstring's fairness note.
        return (slides_pptx.extract_pptx_text(input_path, include_notes=False),
                "python-pptx parser")
    if name == PYPDF_PATH:
        return slides_ocr.extract_pdf_text(pdf_path), "pypdf text layer"
    if name == TESSERACT_PATH:
        if suffix in IMAGE_SUFFIXES:
            return slides_ocr.ocr_image(input_path), "Tesseract OCR engine"
        return _ocr_shared_images(images), "Tesseract OCR engine"
    if name == VISION_PATH:
        text, report = slide_vision.extract_from_images_with_report(
            images, client)
        if report_out is not None:
            report_out.update(report)
        return text, client.model
    raise ValueError(f"Unknown extraction path '{name}'.")


def _save_text(out_dir, slide_set, name, text):
    safe = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(out_dir, f"{stamp}_{slide_set}_{safe}.txt")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return path


def _page_count(input_path, pdf_path, suffix, images, pypdf_ok):
    if images is not None:
        return len(images)
    if pdf_path and pypdf_ok:
        from pypdf import PdfReader
        return len(PdfReader(str(pdf_path)).pages)
    if suffix == slides_pptx.PPTX_SUFFIX and slides_pptx.is_available():
        from pptx import Presentation
        return len(Presentation(str(input_path)).slides)
    return 1


def main(argv=None, client=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True,
                        help="Path to a slide deck (.pptx or .pdf) or a "
                             "slide image.")
    parser.add_argument("--out", default="outputs")
    parser.add_argument("--dpi", type=int, default=config.VISION_DPI)
    parser.add_argument("--max-pages", type=int, default=config.VISION_MAX_PAGES)
    args = parser.parse_args(argv)

    suffix = Path(args.input).suffix.lower()
    slide_set = Path(args.input).stem
    pypdf_ok = slides_ocr.pdf_available()
    ocr_ok = slides_ocr.ocr_available()
    render_ok = slide_vision.is_available()
    pptx_ok = slides_pptx.is_available()
    libreoffice_ok = slides_pptx.libreoffice_available()

    runnable, skipped = build_paths(suffix, pypdf_ok, ocr_ok, render_ok,
                                    pptx_ok=pptx_ok,
                                    libreoffice_ok=libreoffice_ok)
    for name, reason in skipped:
        print(f"Skipping '{name}': {reason}.")
    if suffix == slides_pptx.PPTX_SUFFIX and not libreoffice_ok:
        print("LibreOffice was not found, so only the native pptx text path "
              "can run for this deck. Install LibreOffice, or export the deck "
              "to PDF and pass that PDF, to compare all four paths.")
    if not runnable:
        print(f"No extraction paths can run for '{args.input}'. "
              "Provide a .pptx, .pdf, .png or .jpg and install the needed "
              "extras.")
        return 1

    client = client or LLMClient(model=config.VISION_MODEL)

    # Pre-flight the vision model before any extraction work: page rendering
    # and OCR are slow, so an unpulled model must be reported up front rather
    # than after the other paths have already run. Skipping, not fatal, so the
    # classical paths still produce their comparison rows.
    if VISION_PATH in runnable:
        reachable, message = client.check_model_available()
        if not reachable:
            print(f"Skipping '{VISION_PATH}': {message}")
            runnable = [name for name in runnable if name != VISION_PATH]
            if not runnable:
                print("No extraction paths can run.")
                return 1

    with ExitStack() as stack:
        # A pptx becomes a PDF once, and every PDF-based path then runs
        # unchanged, so nothing downstream needs to know about PowerPoint.
        pdf_path = args.input if suffix == ".pdf" else None
        pdf_paths_wanted = {PYPDF_PATH, TESSERACT_PATH, VISION_PATH}
        if (suffix == slides_pptx.PPTX_SUFFIX
                and pdf_paths_wanted & set(runnable)):
            workspace = stack.enter_context(tempfile.TemporaryDirectory())
            print("Converting the deck to PDF with LibreOffice...")
            pdf_path = slides_pptx.convert_to_pdf(args.input, workspace)

        # Render the pages once and share them between the image-based paths.
        images = None
        needs_render = pdf_path is not None and (
            TESSERACT_PATH in runnable or VISION_PATH in runnable)
        if needs_render:
            images = slide_vision.render_pdf_to_images(
                pdf_path, dpi=args.dpi, max_pages=args.max_pages)
        elif suffix in IMAGE_SUFFIXES and VISION_PATH in runnable:
            images = [slide_vision.encode_image_file(args.input)]

        pages = _page_count(args.input, pdf_path, suffix, images, pypdf_ok)

        os.makedirs(args.out, exist_ok=True)
        rows = []
        for name in runnable:
            print(f"Running extraction path: {name}")
            start = time.perf_counter()
            report = {}
            text, model_used = _run_path(name, args.input, pdf_path, suffix,
                                         images, client, report_out=report)
            elapsed = time.perf_counter() - start
            out_file = _save_text(args.out, slide_set, name, text)
            row = {column: "" for column in CSV_COLUMNS}
            row["Slide set"] = slide_set
            row["Pages"] = pages
            row["Extraction path"] = name
            row["Character count"] = len(text)
            row["Word count"] = len(text.split())
            row["Seconds elapsed"] = f"{elapsed:.2f}"
            row["Model used"] = model_used
            if report:
                unreadable = report.get("unreadable_pages") or []
                truncated = report.get("truncated_pages") or []
                row["Unreadable pages"] = len(unreadable)
                row["Truncated pages"] = len(truncated)
                row["Retries spent"] = report.get("retries", 0)
            row["Output text file"] = out_file
            rows.append(row)
            print(f"  saved {out_file} "
                  f"({elapsed:.2f}s, {len(text.split())} words)")
            note = slide_vision.describe_report(report)
            if note:
                # Printed rather than buried in the sheet, because an
                # extraction with missing pages must not look like a clean run
                # while it is happening.
                print(f"  WARNING: {note}")

    csv_path = os.path.join(args.out, "extractor_comparison.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Comparison sheet written to {csv_path}. "
          "Fill in the score columns by hand.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
