"""Ablation: what does per-slide routing cost, and what does it save?

The routing rule was derived from 111 instrumented slides and says: send a
slide to the vision model when pictures occupy at least 10 per cent of it,
falling back to text yield under 60 words when picture area cannot be read.
Deriving a threshold is not the same as showing it works, and a threshold
evaluated on the run that produced it proves nothing.

This script runs the rule end to end on all five decks and compares what came
out against the all-vision extraction already measured in
outputs/extractors_ctx16k, which is the condition the rule is meant to
improve on.

Three numbers decide whether the rule is worth shipping:

  * **Content retained.** Words extracted under routing, as a proportion of
    words extracted when every slide went to the model. Routing is only
    defensible if it keeps nearly all the content.
  * **Model calls avoided.** The saving, and the reason for doing this.
  * **Time saved.** What the saving is worth in seconds a student waits.

Content retained is the honest headline, and it is reported first, because a
rule that halves the runtime by discarding a fifth of the material is not an
improvement. Note also what word count cannot see: routing a slide to the
text layer loses its diagram description specifically, and a diagram
description is short. A rule could retain 97 per cent of the words while
losing the exact content the vision model exists to produce. The per-deck
routed output is saved so that can be checked by reading rather than assumed.

Usage, from the repository root:

    python run_ablation.py

About ten to fifteen minutes. Run it on a quiet machine.
"""
import argparse
import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path

CORPUS = Path("..") / "Evaluation corpus"
DRAFT_DECK = (Path("..") / "Testing data" / "Naive Bayes test data used"
              / "CM3060 L6.pptx")
BASELINE_ROOT = Path("outputs") / "extractors_ctx16k"

CSV_COLUMNS = [
    "Deck",
    "Content kept by eye (1 to 5)",
    "Notes",
    "pages",
    "vision_pages",
    "text_pages",
    "calls_avoided_pct",
    "routed_words",
    "all_vision_words",
    "content_retained_pct",
    "routed_seconds",
    "all_vision_seconds",
    "time_saved_pct",
    "unreadable_pages",
    "output_file",
]


class Log:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.handle = open(path, "a", encoding="utf-8")

    def __call__(self, message=""):
        stamped = (f"[{datetime.now().strftime('%H:%M:%S')}] {message}"
                   if message else "")
        print(stamped)
        sys.stdout.flush()
        self.handle.write(stamped + "\n")
        self.handle.flush()

    def close(self):
        self.handle.close()


def baseline_for(deck_label):
    """Read the all-vision figures already measured for this deck.

    Returns (words, seconds) or (None, None). Read from the saved sheet
    rather than re-run, because the all-vision condition was measured under
    the same model and settings an hour ago and re-running it would only add
    sampling noise to the comparison.
    """
    path = BASELINE_ROOT / deck_label / "extractor_comparison.csv"
    if not path.is_file():
        return None, None
    with open(path, newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row.get("Extraction path", "").startswith("Vision"):
                try:
                    return (int(row["Word count"]),
                            float(row["Seconds elapsed"]))
                except (KeyError, TypeError, ValueError):
                    return None, None
    return None, None


def discover_decks():
    decks = []
    if DRAFT_DECK.is_file():
        decks.append(("CM3060_L6_draft_deck", DRAFT_DECK))
    if CORPUS.is_dir():
        for child in sorted(CORPUS.iterdir()):
            if not child.is_dir():
                continue
            found = [p for p in sorted(child.iterdir())
                     if p.suffix.lower() in (".pptx", ".pdf")]
            if found:
                decks.append((child.name, found[0]))
    return decks


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="outputs/ablation_routing")
    parser.add_argument("--picture-threshold", type=float, default=None)
    parser.add_argument("--words-threshold", type=int, default=None)
    args = parser.parse_args(argv)

    os.environ.setdefault("LLM_TEMPERATURE", "0")

    import config
    from core.llm_client import LLMClient
    from modules import slide_routing

    out_dir = Path(args.out)
    (out_dir / "routed").mkdir(parents=True, exist_ok=True)
    log = Log(out_dir / "ablation.log")

    log("=" * 68)
    log("Routing ablation")
    log("=" * 68)
    log(f"vision model: {config.VISION_MODEL}")
    log(f"picture area threshold: "
        f"{args.picture_threshold or slide_routing.PICTURE_AREA_THRESHOLD}")
    log(f"text yield threshold  : "
        f"{args.words_threshold or slide_routing.TEXT_WORDS_THRESHOLD}")

    client = LLMClient(model=config.VISION_MODEL)
    ok, message = client.check_model_available()
    if not ok:
        log(f"ABORT: {message}")
        log.close()
        return 1
    log(message)

    decks = discover_decks()
    if not decks:
        log("ABORT: no decks found.")
        log.close()
        return 1

    rows = []
    for label, path in decks:
        log("-" * 68)
        log(f"deck: {label}")
        base_words, base_seconds = baseline_for(label)
        if base_words is None:
            log("  no all-vision baseline recorded for this deck, skipping. "
                "Run run_refresh.py first.")
            continue

        def progress(number, route, reason, seconds):
            log(f"  page {number:>3}: {route:<11} {seconds:>5.1f}s  {reason}")

        started = time.perf_counter()
        try:
            text, report = slide_routing.extract_routed(
                path, client,
                picture_threshold=args.picture_threshold,
                words_threshold=args.words_threshold,
                on_page=progress)
        except Exception as error:
            log(f"  FAILED: {str(error)[:250]}")
            continue
        elapsed = time.perf_counter() - started

        safe = label.replace(" ", "_")
        out_file = out_dir / "routed" / f"{safe}_routed.txt"
        out_file.write_text(text, encoding="utf-8")

        words = len(text.split())
        retained = 100.0 * words / base_words if base_words else 0.0
        saved_time = (100.0 * (1 - elapsed / base_seconds)
                      if base_seconds else 0.0)
        avoided = (100.0 * report["text_pages"] / report["pages"]
                   if report["pages"] else 0.0)

        row = {column: "" for column in CSV_COLUMNS}
        row.update({
            "Deck": label,
            "pages": report["pages"],
            "vision_pages": report["vision_pages"],
            "text_pages": report["text_pages"],
            "calls_avoided_pct": f"{avoided:.0f}",
            "routed_words": words,
            "all_vision_words": base_words,
            "content_retained_pct": f"{retained:.0f}",
            "routed_seconds": f"{elapsed:.1f}",
            "all_vision_seconds": f"{base_seconds:.1f}",
            "time_saved_pct": f"{saved_time:.0f}",
            "unreadable_pages": len(report["unreadable_pages"]),
            "output_file": str(out_file),
        })
        rows.append(row)

        log(f"  {slide_routing.summarise(report)}")
        log(f"  words {words} against {base_words} all-vision "
            f"({retained:.0f} per cent retained)")
        log(f"  {elapsed:.0f}s against {base_seconds:.0f}s "
            f"({saved_time:.0f} per cent faster)")

    csv_path = out_dir / "ablation.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    log("=" * 68)
    log("Summary")
    log("=" * 68)
    log(f"{'deck':<26} {'calls saved':>12} {'content kept':>13} {'faster':>8}")
    for row in rows:
        log(f"{row['Deck']:<26} {row['calls_avoided_pct']:>11}% "
            f"{row['content_retained_pct']:>12}% "
            f"{row['time_saved_pct']:>7}%")
    if rows:
        total_pages = sum(int(r["pages"]) for r in rows)
        total_vision = sum(int(r["vision_pages"]) for r in rows)
        total_words = sum(int(r["routed_words"]) for r in rows)
        total_base = sum(int(r["all_vision_words"]) for r in rows)
        log("")
        log(f"overall: {total_vision} of {total_pages} slides sent to the "
            f"model ({100 * (1 - total_vision / total_pages):.0f} per cent "
            f"avoided), {100 * total_words / total_base:.0f} per cent of the "
            f"content retained")
    log("")
    log(f"sheet : {csv_path}")
    log(f"routed: {out_dir / 'routed'}")
    log("")
    log("Word count cannot tell a lost diagram description from a lost "
        "duplicate line. Read two or three routed pages against their "
        "all-vision counterparts and fill in the manual column before "
        "presenting the retention figure as a quality claim.")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
