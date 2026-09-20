"""Rebuild every vision-model result invalidated by the 4096 token window.

One unattended run. Safe to start and walk away from.

Background: `qwen3-vl:4b` served through Ollama allocates a 4096 token
context window. A rendered slide costs a fixed 2303 tokens of it, leaving
1793 to generate in, and qwen3 reasons before it answers. On a dense slide
the reasoning consumed the whole allowance and the answer never started, so
the call returned an empty string with no error. That happened on 49 of 111
slides in the C3 instrumentation pass and on 15 of 36 pages in the extractor
comparison behind the draft report. Both results have to be produced again.

What this script does, in order:

  1. Builds the derived model `qwen3-vl-ctx16k` from `Modelfile.vision` if it
     does not already exist.

  2. **Gates on proof that the fix works.** Four pages that returned nothing
     before are called again, and every one must come back with a
     finish_reason of "stop" and a non-empty reply. If any still reports
     "length", the script stops immediately and changes nothing else. This
     costs about two minutes and is the whole point of the design: the
     alternative is discovering the fix did not work after two hours of
     generation, which is exactly what happened on the first attempt.

  3. Moves the existing `outputs/c3_routing` aside under a dated name rather
     than deleting it. The faulty run is evidence for the finding and the
     report cites its numbers.

  4. Re-runs the C3 instrumentation pass over the four corpus decks.

  5. Checks the new sheet and reports the empty-reply rate. Anything above a
     few per cent is flagged loudly in the log.

  6. Re-runs the extractor comparison on all four corpus decks and on the
     CM3060 deck used in the draft, each into its own directory so the
     comparison sheets do not overwrite one another.

Every step is logged with a timestamp to `outputs/refresh_run.log` as well as
to the console, and a failure in one deck is recorded and stepped over rather
than ending the run.

The vision model is selected by setting VISION_MODEL in the environment
before the project configuration is imported, so no edit to `.env` is needed
for this run and nothing about the normal configuration is changed.

Usage, from the repository root:

    python run_refresh.py

Expect roughly two to two and a half hours. Run it on a quiet machine.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# qwen3-vl:4b was rejected after the bake-off. It answered 5 of 9 test pages
# against 9 of 9 for this model, could not be stopped reasoning by any of the
# four controls the deployment exposes, and failed non-deterministically: the
# same page at temperature 0 consumed 14641, 4825 and 10014 tokens on three
# consecutive calls. qwen2.5vl:7b finishes cleanly on every page at about 17
# seconds each.
DEFAULT_MODEL = "qwen2.5vl:7b"

# Only used when the chosen model has to be built rather than pulled, which
# was the case for the derived qwen3-vl-ctx16k and is not for this one.
MODELFILE = "Modelfile.vision"

CORPUS = Path("..") / "Evaluation corpus"

# The deck behind the extractor comparison in the draft report. Re-running it
# is the higher priority of the two, because it is evidence for a claim that
# has already been submitted.
DRAFT_DECK = (Path("..") / "Testing data" / "Naive Bayes test data used"
              / "CM3060 L6.pptx")

# Pages that returned nothing under the 4096 token window. Deck 01 page 13
# is the heaviest page in the corpus and deck 04 page 6 is from the deck
# that failed 12 times out of 15, so between them they cover the worst of
# what the corpus contains.
GATE_PAGES = [
    ("01_AI_L8_control", "CM3020 Artificial Intelligence_lecture 8-2.pptx",
     [5, 13, 28]),
    ("04_GP_T7_extreme", "CM2030_GP_PPT7-CourseraWeeks13&14.pptx", [6]),
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

    def rule(self, title=""):
        self("=" * 68)
        if title:
            self(title)
            self("=" * 68)

    def close(self):
        self.handle.close()


def elapsed_text(seconds):
    seconds = int(seconds)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours}h {minutes}m" if hours else f"{minutes}m {seconds}s"


def ensure_model(log, model):
    """Build the derived model if it is not already present.

    Returns True when the model is available afterwards. A missing ollama
    binary or a failed build is reported and stops the run, because every
    later step depends on it.
    """
    binary = shutil.which("ollama")
    if binary is None:
        log("ABORT: the 'ollama' command was not found on PATH.")
        return False
    try:
        listing = subprocess.run([binary, "list"], capture_output=True,
                                 text=True, timeout=60, check=False)
    except (OSError, subprocess.SubprocessError) as error:
        log(f"ABORT: could not run 'ollama list': {error}")
        return False
    if model in (listing.stdout or ""):
        log(f"model '{model}' already exists, not rebuilding")
        return True
    if not Path(MODELFILE).is_file():
        log(f"ABORT: {MODELFILE} not found in the current directory. "
            "Run this from the repository root.")
        return False
    log(f"building '{model}' from {MODELFILE}...")
    try:
        built = subprocess.run([binary, "create", model, "-f", MODELFILE],
                               capture_output=True, text=True, timeout=900,
                               check=False)
    except (OSError, subprocess.SubprocessError) as error:
        log(f"ABORT: 'ollama create' failed to start: {error}")
        return False
    if built.returncode != 0:
        log(f"ABORT: 'ollama create' failed: "
            f"{(built.stderr or built.stdout or '').strip()[:400]}")
        return False
    log(f"built '{model}'")
    return True


def gate(log, config, dpi):
    """Prove the fix works before spending hours on the strength of it.

    Calls pages that previously returned nothing and requires every one to
    finish with reason "stop" and a non-empty reply. Returns True only if
    all of them pass.
    """
    import base64
    import tempfile
    import fitz
    from openai import OpenAI
    from modules import slide_vision, slides_pptx

    client = OpenAI(api_key=config.OPENAI_API_KEY,
                    base_url=config.OPENAI_BASE_URL, timeout=300)
    passed = 0
    attempted = 0

    for folder, filename, pages in GATE_PAGES:
        deck = CORPUS / folder / filename
        if not deck.is_file():
            log(f"  gate deck missing, skipping: {deck}")
            continue
        workspace = tempfile.TemporaryDirectory()
        try:
            pdf_path = Path(slides_pptx.convert_to_pdf(str(deck),
                                                       workspace.name))
            document = fitz.open(str(pdf_path))
            try:
                for number in pages:
                    if number > len(document):
                        continue
                    attempted += 1
                    pixmap = document[number - 1].get_pixmap(dpi=dpi)
                    url = ("data:image/png;base64," + base64.b64encode(
                        pixmap.tobytes("png")).decode("ascii"))
                    started = time.perf_counter()
                    try:
                        reply = client.chat.completions.create(
                            model=config.VISION_MODEL,
                            messages=[{"role": "user", "content": [
                                {"type": "text",
                                 "text": slide_vision.VISION_INSTRUCTION},
                                {"type": "image_url",
                                 "image_url": {"url": url}}]}],
                            temperature=0)
                    except Exception as error:
                        log(f"  {folder} p{number}: CALL FAILED: "
                            f"{str(error)[:160]}")
                        continue
                    took = time.perf_counter() - started
                    choice = reply.choices[0]
                    words = len((choice.message.content or "").split())
                    usage = getattr(reply, "usage", None)
                    completion = getattr(usage, "completion_tokens", "?")
                    ok = choice.finish_reason == "stop" and words > 0
                    log(f"  {folder} p{number}: {took:>5.1f}s "
                        f"finish={choice.finish_reason} "
                        f"completion_tok={completion} words={words} "
                        f"{'PASS' if ok else 'FAIL'}")
                    if ok:
                        passed += 1
            finally:
                document.close()
        except Exception as error:
            log(f"  gate error on {folder}: {str(error)[:200]}")
        finally:
            workspace.cleanup()

    log(f"  gate result: {passed} of {attempted} pages passed")
    return attempted > 0 and passed == attempted


def archive(log, path):
    """Move a directory aside under a dated name instead of deleting it.

    The faulty run is the evidence behind the finding and its numbers are
    quoted in the report, so it is kept.
    """
    target = Path(path)
    if not target.exists():
        return
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    moved = target.with_name(f"{target.name}_pre_ctx_fix_{stamp}")
    try:
        target.rename(moved)
        log(f"  moved previous results to {moved}")
    except OSError as error:
        log(f"  WARNING: could not move {target} ({error}). "
            "The pass will resume over the old rows instead of starting "
            "fresh, which is not what is wanted here.")


def summarise_c3(log, csv_path):
    import csv
    if not Path(csv_path).is_file():
        log("  WARNING: no sheet was produced.")
        return
    with open(csv_path, newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        log("  WARNING: the sheet is empty.")
        return
    empty = sum(1 for r in rows if int(r.get("vision_words") or 0) == 0)
    errors = sum(1 for r in rows if (r.get("error") or "").strip())
    rate = 100.0 * empty / len(rows)
    log(f"  {len(rows)} pages, {errors} errors, {empty} empty replies "
        f"({rate:.1f} per cent)")
    if rate > 5.0:
        log("  *** STILL FAILING: the empty-reply rate is too high. Do not")
        log("  *** use this sheet. Report the rate before re-running.")
    else:
        log("  empty-reply rate is acceptable, the sheet is usable")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vision-model", default=DEFAULT_MODEL)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--skip-extractors", action="store_true",
                        help="Run only the C3 pass.")
    args = parser.parse_args(argv)

    # Set the model and the reproducibility control before the project
    # configuration is imported, because config.py reads the environment at
    # import time. This is why the imports below sit inside main().
    os.environ["VISION_MODEL"] = args.vision_model
    os.environ.setdefault("LLM_TEMPERATURE", "0")

    log = Log(Path("outputs") / "refresh_run.log")
    started = time.time()
    log.rule("Vision refresh run")
    log(f"vision model: {args.vision_model}   dpi: {args.dpi}")

    if not ensure_model(log, args.vision_model):
        log.close()
        return 1

    import config
    log(f"endpoint: {config.OPENAI_BASE_URL}")
    log(f"config.VISION_MODEL resolved to: {config.VISION_MODEL}")
    if config.VISION_MODEL != args.vision_model:
        log("ABORT: the configuration did not pick up the model. Something "
            "in .env is overriding it.")
        log.close()
        return 1

    log.rule("Step 1 of 3: proving the fix works")
    log("Calling pages that previously returned nothing. Every one must "
        "finish with reason 'stop'.")
    if not gate(log, config, args.dpi):
        log("")
        log("ABORT: the gate did not pass, so nothing else has been run and")
        log("nothing has been moved or overwritten. The context window fix")
        log("is not taking effect. Send this log back before re-running.")
        log.close()
        return 1
    log("gate passed, continuing")

    log.rule("Step 2 of 3: C3 instrumentation pass")
    archive(log, Path("outputs") / "c3_routing")
    from evaluation import route_instrumentation
    try:
        route_instrumentation.main([
            "--corpus", str(CORPUS),
            "--out", "outputs/c3_routing",
            "--dpi", str(args.dpi),
        ])
    except Exception as error:
        log(f"  C3 pass ended with an error: {str(error)[:300]}")
    summarise_c3(log, Path("outputs") / "c3_routing" / "routing_features.csv")

    if args.skip_extractors:
        log.rule("Skipping the extractor comparison as requested")
        log(f"total elapsed {elapsed_text(time.time() - started)}")
        log.close()
        return 0

    log.rule("Step 3 of 3: extractor comparison")
    from evaluation import compare_extractors

    decks = []
    if DRAFT_DECK.is_file():
        # First, because it is the one that backs a claim already submitted.
        decks.append(("CM3060_L6_draft_deck", DRAFT_DECK))
    else:
        log(f"  NOTE: the draft deck was not found at {DRAFT_DECK}. The "
            "corpus decks will still be run.")
    if CORPUS.is_dir():
        for child in sorted(CORPUS.iterdir()):
            if not child.is_dir():
                continue
            found = [p for p in sorted(child.iterdir())
                     if p.suffix.lower() in (".pptx", ".pdf")]
            if found:
                decks.append((child.name, found[0]))

    for label, path in decks:
        out_dir = Path("outputs") / "extractors_ctx16k" / label
        out_dir.mkdir(parents=True, exist_ok=True)
        log("-" * 68)
        log(f"deck: {label}")
        deck_started = time.time()
        try:
            compare_extractors.main([
                "--input", str(path),
                "--out", str(out_dir),
                "--dpi", str(args.dpi),
                # Explicit and generous. The configuration default is 20,
                # and three of these decks are longer than that, so relying
                # on it would silently drop their later slides.
                "--max-pages", "200",
            ])
            log(f"  done in {elapsed_text(time.time() - deck_started)}")
        except Exception as error:
            log(f"  FAILED: {str(error)[:300]}")

    log.rule("Finished")
    log(f"total elapsed {elapsed_text(time.time() - started)}")
    log("C3 sheet      : outputs/c3_routing/routing_features.csv")
    log("extractor runs: outputs/extractors_ctx16k/<deck>/"
        "extractor_comparison.csv")
    log.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
