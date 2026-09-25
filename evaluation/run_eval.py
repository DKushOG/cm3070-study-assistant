"""Run the pipeline over every input mode and write a scoring sheet.

Uses the sample files, saves every output and writes a CSV with one row per
run. The timing columns are filled in automatically and the scoring columns
are left blank for marking by hand. Pointing --samples at a different folder
runs the same evaluation on another topic.

--repeats runs each mode more than once and adds a run column, so a mode can
be reported as a mean. With a single run per mode, a difference between modes
cannot be separated from run-to-run sampling noise. --temperature and --seed
pass the sampling controls through to the client, and the model, temperature,
seed and slide source are written into the CSV, so a results file still
explains itself months later.

The quiz validation columns (questions_found, returned_five, attempts,
validation_passed) replace a column that used to be filled in by hand, which
turns an observation into a measurement.

A mode whose source files are missing is skipped with a warning rather than
crashing, so the runner still works while a topic folder is being put
together. The client can be injected for testing.

Usage:
    python -m evaluation.run_eval --samples samples --out outputs --repeats 3
"""
import argparse
import csv
import os

import config
from core.llm_client import LLMClient
from core.quiz_validation import (GENERATED_NOTES_FAULT,
                                  INVALID_BASIS_FAULT)
from core.orchestrator import run_pipeline, save_result
from evaluation.rubric import CRITERIA

SAMPLE_FILES = {
    "transcript": "transcript_sample.txt",
    "slide_text": "slide_text_sample.txt",
    "notes": "notes_sample.txt",
}


def _blank_if_none(value):
    """Render an unset run setting as an empty CSV cell rather than the text
    'None', which would be indistinguishable from a real value in a
    spreadsheet."""
    return "" if value is None else value


def load(path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read()
    return None


def build_modes(transcript, slide_text, notes):
    """Return the runnable modes and the names of any skipped ones.

    A mode is runnable only if every source it needs has content.
    """
    candidates = [
        ("Transcript only", dict(transcript=transcript)),
        ("Slide text only", dict(slide_text=slide_text)),
        ("Combined transcript and slide text",
         dict(transcript=transcript, slide_text=slide_text)),
        ("Transcript, slide text and notes",
         dict(transcript=transcript, slide_text=slide_text, notes=notes)),
    ]
    runnable = []
    skipped = []
    for name, kwargs in candidates:
        if all(value and value.strip() for value in kwargs.values()):
            runnable.append((name, kwargs))
        else:
            skipped.append(name)
    return runnable, skipped


def main(argv=None, client=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", default="samples")
    parser.add_argument("--out", default="outputs")
    parser.add_argument(
        "--repeats", type=int, default=1,
        help="Runs per mode. More than one lets a mean be reported instead "
             "of a single sample.")
    parser.add_argument(
        "--temperature", type=float, default=None,
        help="Sampling temperature. Unset means the model default, exactly "
             "as in the baseline runs.")
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Sampling seed for repeatable runs. Unset means no seed.")
    parser.add_argument(
        "--slide-source", default=None,
        help="Label recording which extractor produced the slide text "
             "sample, e.g. 'Tesseract OCR' or 'Vision model', so runs fed by "
             "different extractors can be compared.")
    args = parser.parse_args(argv)

    sources = {key: load(os.path.join(args.samples, filename))
               for key, filename in SAMPLE_FILES.items()}
    runnable, skipped = build_modes(sources["transcript"],
                                    sources["slide_text"],
                                    sources["notes"])
    for name in skipped:
        print(f"Skipping mode '{name}': a source file it needs is missing "
              f"or empty in {args.samples}.")
    if not runnable:
        expected = ", ".join(SAMPLE_FILES.values())
        print(f"No input modes can run. Expected sample files in "
              f"'{args.samples}': {expected}")
        return 1

    client = client or LLMClient(temperature=args.temperature, seed=args.seed)
    rows = []
    for run_number in range(1, max(1, args.repeats) + 1):
        for name, kwargs in runnable:
            print(f"Running mode: {name} (run {run_number}"
                  f" of {max(1, args.repeats)})")
            result = run_pipeline(
                client, max_words=config.MAX_CONTEXT_WORDS,
                slide_source=args.slide_source,
                quiz_max_attempts=config.QUIZ_MAX_ATTEMPTS, **kwargs)
            path = save_result(result, output_dir=args.out)
            validation = result.quiz_validation
            row = {
                "mode": result.mode,
                "run": run_number,
                "notes_seconds": f"{result.notes_seconds:.2f}",
                "quiz_seconds": f"{result.quiz_seconds:.2f}",
                "total_seconds": f"{result.total_seconds:.2f}",
                "model": _blank_if_none(result.model),
                "temperature": _blank_if_none(result.temperature),
                "seed": _blank_if_none(result.seed),
                "slide_source": _blank_if_none(result.slide_source),
                # Measured, not hand-filled: these replace the manual
                # "Returned five questions" column in the workbook.
                "questions_found": (
                    "" if validation is None else validation.questions_found),
                "returned_five": (
                    "" if validation is None else validation.returned_five),
                "attempts": result.quiz_attempts,
                "validation_passed": (
                    "" if validation is None else validation.passed),
                "output_file": path,
            }
            # Rubric columns stay blank for manual scoring, deliberately.
            for key, _description in CRITERIA:
                row[key] = ""
            # Appended strictly after every existing column, including the
            # rubric ones, so no column already read by the evaluation
            # workbook changes position.
            row["invalid_source_basis"] = (
                "" if validation is None
                else validation.count_faults(INVALID_BASIS_FAULT))
            row["generated_notes_basis"] = (
                "" if validation is None
                else validation.count_faults(GENERATED_NOTES_FAULT))
            row["shape_warnings"] = (
                "" if validation is None else validation.shape_warning_count)
            row["quiz_seeds"] = ",".join(str(s) for s in result.quiz_seeds)
            # Which generation path produced this quiz. Appended last,
            # like the columns above it, so nothing already read by the
            # evaluation workbook shifts position. Without it a results
            # file cannot be attributed to a path once the folders are
            # merged, and the before-and-after comparison is the whole
            # point of the structured path existing.
            row["quiz_structured"] = result.quiz_structured
            rows.append(row)
            print(f"  saved {path} ({result.total_seconds:.2f}s)")

    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, "evaluation_scores.csv")
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Scoring sheet written to {csv_path}. "
          "Fill in the rubric columns by hand.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
