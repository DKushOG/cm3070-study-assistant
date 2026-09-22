"""C3 instrumentation pass: per-slide features against vision-model outcome.

The system currently applies one extraction path to a whole deck. Every page
of a slide deck is either read cheaply as text or sent to the vision-language
model, and the choice is made once for the deck rather than once per slide.
That is wasteful in one direction and lossy in the other. A text-dense slide
sent to the vision model costs a model call to recover text that pypdf already
had. A slide that is a single unlabelled diagram, read by the text path,
yields almost nothing.

C3 is the proposal that the choice should be made per slide, by a rule using
features that can be computed before any model call is made. This script does
not implement that rule. It gathers the evidence needed to derive one, which
is a deliberately separate step: deriving a threshold from the same run that
implements it would be circular.

What it records, for every page of every deck in the corpus:

  * Cheap features, all obtainable without a model call. Text yield from the
    PDF text layer, the number of text blocks, the count and page area of
    placed images, the number of vector drawing operations, and the
    proportion of the rendered page that is not blank. For a PowerPoint
    deck, the native shape counts and picture area are recorded as well.

  * The outcome of calling the vision model on that page: how long it took,
    how much it returned, and how much of what it returned was not already
    available from the text layer.

The outcome measure is deliberately a proxy and is named as one. "Novel
content words" counts content words in the vision output that do not appear
in the page's text layer. It rewards a model that described a diagram the
text layer could not see, and it does not reward one that re-transcribed
text pypdf already had. It also cannot distinguish a genuine description
from a paraphrase of existing text, which is why a manual column is provided
and why a sample should be scored by hand before any threshold is fixed.

Fairness follows evaluation/compare_extractors.py: a PowerPoint deck is
converted to PDF once with LibreOffice, after which every deck is treated
identically, and each page is rendered once and shared between the feature
pass and the model call.

Designed to run unattended. Rows are written and flushed as each page
finishes, a page that fails is recorded with its error rather than ending the
run, and re-running against an existing output directory resumes where it
stopped instead of repeating work.

Usage:
    # check the feature pass works, no model calls, takes under a minute
    python -m evaluation.route_instrumentation --corpus "..\\Evaluation corpus" ^
        --out outputs/c3_routing --dry-run

    # the real pass
    python -m evaluation.route_instrumentation --corpus "..\\Evaluation corpus" ^
        --out outputs/c3_routing
"""
import argparse
import base64
import csv
import os
import re
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import config
from core.llm_client import LLMClient
from modules import slide_routing, slide_vision, slides_pptx

DECK_SUFFIXES = (".pptx", ".pdf")

# Resolution for the blankness measurement only. Deliberately far lower than
# the extraction DPI: this is a whole-page statistic, so rendering it at
# extraction quality would cost time for no extra information.
INK_DPI = 40

# A pixel darker than this counts as marked. Slides are overwhelmingly light
# on white, and a threshold near the top of the range catches faint grey
# gridlines and pale backgrounds that a stricter one would miss.
INK_THRESHOLD = 245

# Words carrying no topic content, excluded from the novelty measure so that
# a vision output is not credited for producing articles and prepositions the
# text layer happened not to contain.
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "can", "could", "did", "do", "does", "for", "from", "had", "has", "have",
    "he", "her", "his", "how", "i", "if", "in", "into", "is", "it", "its",
    "may", "might", "must", "no", "not", "of", "on", "or", "our", "out",
    "shall", "she", "should", "so", "such", "than", "that", "the", "their",
    "them", "then", "there", "these", "they", "this", "those", "to", "up",
    "was", "we", "were", "what", "when", "which", "while", "who", "will",
    "with", "would", "you", "your",
    # Vision-model scaffolding. The instruction asks for a transcription and
    # then a description, and the model tends to label those sections. Those
    # labels are not slide content and must not count as novel.
    "slide", "image", "text", "shows", "showing", "appears", "title",
    "transcription", "description", "diagram", "figure", "visible",
}

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

# Manual columns first, matching the house style of the other harnesses, so
# the sheet can be hand-scored without rearranging it. Left blank here.
MANUAL_COLUMNS = [
    "Deck",
    "Page",
    "Vision earned its cost (1 to 5)",
    "Notes",
]

FEATURE_COLUMNS = [
    "source_file",
    "source_type",
    "pages_in_deck",
    "text_layer_words",
    "text_layer_chars",
    "text_blocks",
    "placed_images",
    "image_area_ratio",
    "drawing_ops",
    "ink_ratio",
    "pptx_picture_shapes",
    "pptx_chart_shapes",
    "pptx_table_shapes",
    "pptx_group_shapes",
    "pptx_text_shapes",
    "pptx_picture_area_ratio",
    "pptx_native_words",
]

OUTCOME_COLUMNS = [
    "vision_seconds",
    "vision_words",
    "vision_chars",
    "novel_content_words",
    "text_layer_content_words",
    "novel_ratio",
    "vision_model",
    # Recorded because the first run of this pass reported 49 of 111 pages as
    # measured when the model had returned an empty string for each of them.
    # A finish reason of "length" means the reply was cut off, which no
    # amount of reading the returned text can reveal on its own.
    "finish_reason",
    "vision_output_file",
    "error",
]

CSV_COLUMNS = MANUAL_COLUMNS + FEATURE_COLUMNS + OUTCOME_COLUMNS

CSV_NAME = "routing_features.csv"

# A feature-only pass writes to its own sheet. Sharing one file would be a
# trap: dry-run rows carry no error, so the resume check would treat every
# page as already recorded and the real pass would skip the entire corpus.
DRY_RUN_CSV_NAME = "routing_features.dryrun.csv"

LOG_NAME = "routing_run.log"


# --------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------

def discover_decks(corpus_root, extra_decks):
    """Return [(deck_label, path)] for every deck to instrument.

    A corpus directory is searched one level deep, which matches the layout
    of the evaluation corpus: one folder per deck, the deck file inside it,
    recordings and transcripts in subfolders that are not slide decks and are
    therefore not picked up. The folder name becomes the label, because
    "02_AI_L6_mixed" identifies a deck far better than the filename the
    module happened to ship with.
    """
    decks = []
    if corpus_root:
        root = Path(corpus_root)
        if not root.is_dir():
            raise SystemExit(f"Corpus directory not found: {root}")
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            found = [p for p in sorted(child.iterdir())
                     if p.suffix.lower() in DECK_SUFFIXES]
            if not found:
                continue
            if len(found) > 1:
                print(f"  note: {child.name} holds {len(found)} deck files, "
                      f"using {found[0].name}")
            decks.append((child.name, found[0]))
    for item in extra_decks or []:
        path = Path(item)
        if not path.is_file():
            raise SystemExit(f"Deck not found: {path}")
        decks.append((path.stem, path))
    return decks


# --------------------------------------------------------------------------
# Cheap features, no model call
# --------------------------------------------------------------------------

def ink_ratio(page):
    """Proportion of the rendered page that is not blank.

    Rendered in greyscale at low resolution and thresholded. The translate
    table turns the pixel buffer into one byte per pixel, 1 for marked and 0
    for blank, so the count runs in C rather than in a Python loop. At 147
    pages that is the difference between seconds and minutes.
    """
    import fitz
    pixmap = page.get_pixmap(dpi=INK_DPI, colorspace=fitz.csGRAY)
    table = bytes(1 if value < INK_THRESHOLD else 0 for value in range(256))
    marked = pixmap.samples.translate(table).count(1)
    total = pixmap.width * pixmap.height
    return (marked / total) if total else 0.0


def placed_image_features(page):
    """Count and total area of images actually placed on this page.

    Deliberately `get_image_info` rather than `get_images`. The latter lists
    the image XObjects in the page's resource dictionary, and a PDF produced
    by LibreOffice from a PowerPoint deck gives every page the same inherited
    resource list. The first version of this script used it and every page of
    a converted deck reported an identical count: 18 for the first deck, 32
    for the second, 24 for the fourth. The number was not a per-page
    measurement at all, and because it was plausible it would have been
    reported as one.

    `get_image_info` returns only images with a placement on the page, each
    with a bounding box, so the count is real and an area can be derived.
    The area is the more useful of the two: a slide given over to one large
    figure is the case where the text path has least to offer.

    Boxes are clipped to the page and summed, so overlapping images
    overstate coverage. The total is capped at 1.0 for that reason, and the
    feature should be read as "how much of the page is given to imagery",
    not as an exact non-overlapping area.
    """
    import fitz
    try:
        infos = page.get_image_info()
    except Exception:
        return 0, 0.0
    rect = page.rect
    page_area = float(rect.width) * float(rect.height)
    covered = 0.0
    count = 0
    for info in infos:
        box = info.get("bbox")
        if not box:
            continue
        try:
            clipped = fitz.Rect(box) & rect
        except Exception:
            continue
        if clipped.is_empty:
            continue
        count += 1
        covered += float(clipped.width) * float(clipped.height)
    ratio = (covered / page_area) if page_area else 0.0
    return count, round(min(ratio, 1.0), 4)


def pdf_page_features(page):
    """Features available from the PDF itself, for any deck.

    drawing_ops counts vector drawing operations, which is how a chart or a
    flow diagram drawn as lines and shapes registers. A photograph registers
    instead in placed_images and image_area_ratio. A slide can be visually
    heavy through either route, so both are recorded rather than combined.

    Note when reading the sheet: drawing_ops has a floor of about 2 on most
    decks, because a background rectangle and a border are themselves
    drawing operations. The informative signal is the excess above that
    floor, not the raw count.
    """
    text = page.get_text("text") or ""
    blocks = page.get_text("blocks") or []
    try:
        drawings = page.get_drawings()
    except Exception:
        drawings = []
    image_count, image_area = placed_image_features(page)
    return {
        "text_layer_words": len(text.split()),
        "text_layer_chars": len(text),
        "text_blocks": len(blocks),
        "placed_images": image_count,
        "image_area_ratio": image_area,
        "drawing_ops": len(drawings),
        "ink_ratio": round(ink_ratio(page), 4),
        "_text": text,
    }


def pptx_page_features(deck_path):
    """Per-slide shape counts and picture area from the PowerPoint itself.

    Returns {slide_number: {...}} or {} when the deck is not a pptx or
    python-pptx is unavailable. These features are the ones a routing rule
    could use most cheaply of all, because reading them needs no rendering at
    all, but they exist only for PowerPoint input. That asymmetry is exactly
    why the PDF features above are recorded for every deck: a rule that works
    only for pptx would not serve a student who uploads a PDF.
    """
    if Path(deck_path).suffix.lower() != slides_pptx.PPTX_SUFFIX:
        return {}
    if not slides_pptx.is_available():
        return {}
    from pptx import Presentation
    from pptx.enum.shapes import MSO_SHAPE_TYPE

    presentation = Presentation(str(deck_path))
    slide_area = float(presentation.slide_width) * float(
        presentation.slide_height)
    result = {}
    for number, slide in enumerate(presentation.slides, start=1):
        pictures = charts = tables = groups = texts = 0
        picture_area = 0.0
        for shape in slide.shapes:
            if getattr(shape, "has_chart", False):
                charts += 1
            elif getattr(shape, "has_table", False):
                tables += 1
            elif shape.shape_type == MSO_SHAPE_TYPE.GROUP:
                groups += 1
            elif slide_routing.is_picture_shape(shape):
                # Uses the shipped rule's own test, so a picture held in a
                # content placeholder is counted here exactly as the router
                # counts it.
                pictures += 1
                try:
                    picture_area += float(shape.width) * float(shape.height)
                except (TypeError, ValueError):
                    pass
            elif getattr(shape, "has_text_frame", False):
                if (shape.text_frame.text or "").strip():
                    texts += 1
        words = 0
        for shape in slide.shapes:
            if getattr(shape, "has_text_frame", False):
                words += len((shape.text_frame.text or "").split())
        result[number] = {
            "pptx_picture_shapes": pictures,
            "pptx_chart_shapes": charts,
            "pptx_table_shapes": tables,
            "pptx_group_shapes": groups,
            "pptx_text_shapes": texts,
            "pptx_picture_area_ratio": (
                round(picture_area / slide_area, 4) if slide_area else ""),
            "pptx_native_words": words,
        }
    return result


# --------------------------------------------------------------------------
# Outcome measure
# --------------------------------------------------------------------------

def content_tokens(text):
    """Lowercase content words, stopwords and scaffolding removed."""
    return {token for token in TOKEN_PATTERN.findall((text or "").lower())
            if token not in STOPWORDS and len(token) > 2}


def novelty(vision_text, layer_text):
    """How much of the vision output was not already in the text layer.

    Set difference on content words rather than a sequence comparison,
    because the vision model rewrites reading order freely and an order
    sensitive measure would score honest transcription as novel. The
    limitation stands: a paraphrase of existing text still counts as novel
    under this measure, which is why the manual column exists.
    """
    vision = content_tokens(vision_text)
    layer = content_tokens(layer_text)
    novel = vision - layer
    return len(novel), len(layer), (
        round(len(novel) / len(vision), 4) if vision else "")


# --------------------------------------------------------------------------
# Run bookkeeping
# --------------------------------------------------------------------------

def load_done(csv_path):
    """Return the (deck, page) pairs already recorded, for resuming.

    A row whose error column is filled is treated as not done, so a page that
    failed because the endpoint was briefly unreachable is retried on the
    next run rather than being permanently skipped.
    """
    done = set()
    if not os.path.isfile(csv_path):
        return done
    with open(csv_path, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if (row.get("error") or "").strip():
                continue
            try:
                done.add((row["Deck"], int(row["Page"])))
            except (KeyError, TypeError, ValueError):
                continue
    return done


class Log:
    """Print to the console and append to a file at the same time.

    An unattended run that only prints leaves nothing behind to diagnose in
    the morning, and one that only writes to a file cannot be watched while
    it starts. Both, flushed every line, so a hard kill loses nothing.
    """

    def __init__(self, path):
        self.handle = open(path, "a", encoding="utf-8")

    def __call__(self, message):
        stamped = f"[{datetime.now().strftime('%H:%M:%S')}] {message}"
        print(stamped)
        sys.stdout.flush()
        self.handle.write(stamped + "\n")
        self.handle.flush()

    def close(self):
        self.handle.close()


def save_vision_text(out_dir, deck, page, text):
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", deck)
    path = os.path.join(out_dir, "vision", f"{safe}_p{page:03d}.txt")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)
    return path


def format_duration(seconds):
    seconds = int(seconds)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m {seconds}s"


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main(argv=None, client=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus",
                        help="Directory holding one folder per deck.")
    parser.add_argument("--deck", action="append", dest="decks",
                        help="An additional deck file. Repeatable.")
    parser.add_argument("--out", default="outputs/c3_routing")
    parser.add_argument("--dpi", type=int, default=config.VISION_DPI,
                        help="Render resolution for the vision model.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute features only, make no model calls.")
    parser.add_argument("--max-pages", type=int, default=0,
                        help="Cap pages per deck. 0 means no cap. The "
                             "config default of 20 is deliberately not used "
                             "here, because several corpus decks are longer "
                             "than that and a silent cap would drop the "
                             "later slides from the sample.")
    args = parser.parse_args(argv)

    if not args.corpus and not args.decks:
        raise SystemExit("Give --corpus, or at least one --deck.")

    try:
        import fitz  # noqa: F401
    except ImportError:
        raise SystemExit(
            "PyMuPDF is required. Install it with: pip install pymupdf")

    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(
        args.out, DRY_RUN_CSV_NAME if args.dry_run else CSV_NAME)
    log = Log(os.path.join(args.out, LOG_NAME))

    decks = discover_decks(args.corpus, args.decks)
    if not decks:
        raise SystemExit("No decks found.")

    log("=" * 64)
    log(f"C3 instrumentation pass, {'features only' if args.dry_run else 'with vision calls'}")
    log(f"decks: {len(decks)}  dpi: {args.dpi}  out: {args.out}")

    done = load_done(csv_path)
    if done:
        log(f"resuming: {len(done)} pages already recorded, skipping those")

    if not args.dry_run:
        client = client or LLMClient(model=config.VISION_MODEL)
        ok, message = client.check_model_available()
        if not ok:
            log(f"ABORT: {message}")
            log.close()
            return 1
        log(message)
    else:
        client = None

    write_header = not os.path.isfile(csv_path)
    handle = open(csv_path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
    if write_header:
        writer.writeheader()
        handle.flush()

    started = time.time()
    pages_done = 0
    pages_failed = 0

    try:
        for deck_label, deck_path in decks:
            suffix = deck_path.suffix.lower()
            log("-" * 64)
            log(f"deck: {deck_label}  ({deck_path.name})")

            workspace = None
            pdf_path = deck_path
            if suffix == slides_pptx.PPTX_SUFFIX:
                if not slides_pptx.libreoffice_available():
                    log("  SKIPPED: LibreOffice not found, cannot render "
                        "this PowerPoint deck. Export it to PDF and pass the "
                        "PDF with --deck.")
                    continue
                workspace = tempfile.TemporaryDirectory()
                log("  converting to PDF with LibreOffice...")
                try:
                    pdf_path = Path(slides_pptx.convert_to_pdf(
                        str(deck_path), workspace.name))
                except Exception as error:
                    log(f"  SKIPPED: conversion failed: {error}")
                    workspace.cleanup()
                    continue

            try:
                shape_features = pptx_page_features(deck_path)
            except Exception as error:
                log(f"  note: pptx shape features unavailable ({error})")
                shape_features = {}

            try:
                document = fitz.open(str(pdf_path))
            except Exception as error:
                log(f"  SKIPPED: could not open PDF: {error}")
                if workspace:
                    workspace.cleanup()
                continue

            total_pages = len(document)
            limit = total_pages if args.max_pages <= 0 else min(
                total_pages, args.max_pages)
            log(f"  {total_pages} pages, instrumenting {limit}")

            for index in range(limit):
                page_number = index + 1
                if (deck_label, page_number) in done:
                    continue

                row = {column: "" for column in CSV_COLUMNS}
                row["Deck"] = deck_label
                row["Page"] = page_number
                row["source_file"] = deck_path.name
                row["source_type"] = suffix.lstrip(".")
                row["pages_in_deck"] = total_pages

                try:
                    page = document[index]
                    features = pdf_page_features(page)
                    layer_text = features.pop("_text")
                    row.update(features)
                    row.update(shape_features.get(page_number, {}))

                    if args.dry_run:
                        log(f"  page {page_number:>3}: "
                            f"{row['text_layer_words']:>4} words, "
                            f"ink {row['ink_ratio']:.3f}, "
                            f"img area {row['image_area_ratio']:.3f} "
                            f"({row['placed_images']}), "
                            f"{row['drawing_ops']} drawings")
                    else:
                        pixmap = page.get_pixmap(dpi=args.dpi)
                        encoded = base64.b64encode(
                            pixmap.tobytes("png")).decode("ascii")
                        url = f"data:image/png;base64,{encoded}"

                        call_started = time.perf_counter()
                        # A token cap and the finish reason, both for the same
                        # reason: a vision model that loses its way generates
                        # until the window is exhausted and returns nothing,
                        # and without the finish reason that is
                        # indistinguishable from a genuinely blank slide.
                        vision_text, meta = client.chat_with_images(
                            slide_vision.VISION_INSTRUCTION, [url],
                            max_tokens=config.VISION_NUM_PREDICT or None,
                            with_meta=True)
                        elapsed = time.perf_counter() - call_started
                        row["finish_reason"] = (meta or {}).get(
                            "finish_reason", "")

                        novel, layer_count, ratio = novelty(
                            vision_text, layer_text)
                        row["vision_seconds"] = f"{elapsed:.2f}"
                        row["vision_words"] = len(vision_text.split())
                        row["vision_chars"] = len(vision_text)
                        row["novel_content_words"] = novel
                        row["text_layer_content_words"] = layer_count
                        row["novel_ratio"] = ratio
                        row["vision_model"] = client.model
                        row["vision_output_file"] = save_vision_text(
                            args.out, deck_label, page_number, vision_text)
                        flag = ""
                        if row["finish_reason"] == "length":
                            flag = "  CUT OFF"
                        elif not row["vision_words"]:
                            flag = "  EMPTY"
                        log(f"  page {page_number:>3}/{limit}: "
                            f"{elapsed:>6.1f}s  "
                            f"layer {row['text_layer_words']:>4}w  "
                            f"vision {row['vision_words']:>4}w  "
                            f"novel {ratio}{flag}")
                    pages_done += 1

                except KeyboardInterrupt:
                    raise
                except Exception as error:
                    # One bad page must not end an overnight run. The error
                    # goes in the row, and load_done treats a row carrying an
                    # error as unfinished, so re-running retries it.
                    row["error"] = str(error)[:300]
                    pages_failed += 1
                    log(f"  page {page_number:>3}: FAILED: {error}")

                writer.writerow(row)
                handle.flush()

                if pages_done and not args.dry_run and pages_done % 10 == 0:
                    rate = (time.time() - started) / pages_done
                    log(f"  ... {pages_done} pages in "
                        f"{format_duration(time.time() - started)}, "
                        f"{rate:.0f}s per page")

            document.close()
            if workspace:
                workspace.cleanup()

    except KeyboardInterrupt:
        log("interrupted by user. Re-run the same command to resume.")
    finally:
        handle.close()

    total = time.time() - started
    log("=" * 64)
    log(f"done: {pages_done} pages recorded, {pages_failed} failed, "
        f"{format_duration(total)} elapsed")
    log(f"sheet: {csv_path}")
    if pages_failed:
        log("Re-run the same command to retry the failed pages.")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
