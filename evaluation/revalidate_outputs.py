"""Check saved runs again with the current quiz validator.

The validator was corrected after the first real runs had been saved, so the
verdicts written into those files came from the older parser and report fewer
faults than the runs actually had. Regenerating the runs would cost model time
and would not produce the same output anyway, and the saved files are evidence
in their own right. This tool re-reads them and applies the current validator,
so a measurement taken before the fix can be restated instead of thrown away.

It reads the quiz section only, which is everything between the
"=== QUIZ QUESTIONS ===" marker and the timing lines that close the file.
Parsing allows for the metadata header, for CRLF line endings written on
Windows, and for a file that was cut short or has no marker at all because a
run was interrupted. A file that cannot be read is reported and skipped rather
than stopping the batch, as evaluation/run_eval.py does for missing sources.

The CSV is written for use in the evaluation workbook, and the summary line
gives the share of runs whose quiz passed validation.

Usage:
    python -m evaluation.revalidate_outputs --outputs outputs
"""
import argparse
import csv
import os

from core.quiz_validation import (EMPTY_FAULT, GENERATED_NOTES_FAULT,
                                  INVALID_BASIS_FAULT, MISSING_FAULT,
                                  PLACEHOLDER_FAULT, validate_quiz)

QUIZ_MARKER = "=== QUIZ QUESTIONS ==="
# The timing lines close the saved file, so the quiz section ends at whichever
# of them appears first.
TIMING_KEYS = ("notes_seconds=", "quiz_seconds=", "total_seconds=")
MODE_PREFIX = "MODE:"

CSV_COLUMNS = [
    "File",
    "Mode",
    "Questions found",
    "Returned five",
    "Validation passed",
    "Missing labels",
    "Empty values",
    "Placeholder echoes",
    "Faults",
    # Appended after the existing columns so the evaluation workbook's column
    # positions are unchanged.
    "Invalid source basis",
    "Generated notes basis",
    "Shape warnings",
    "Warnings",
]


def read_text(path):
    """Read a saved run, normalising Windows line endings."""
    with open(path, encoding="utf-8", errors="replace") as handle:
        return handle.read().replace("\r\n", "\n").replace("\r", "\n")


def extract_quiz(text):
    """Return the quiz section of a saved run, or None if it has none.

    None rather than an empty string, so "this file has no quiz section" is
    distinguishable from "the quiz section was empty", which are different
    facts about the run.
    """
    if QUIZ_MARKER not in text:
        return None
    section = text.split(QUIZ_MARKER, 1)[1]
    cut = len(section)
    for key in TIMING_KEYS:
        for candidate in ("\n" + key, key):
            position = section.find(candidate)
            if position != -1:
                cut = min(cut, position)
            break
    return section[:cut].strip()


def extract_mode(text):
    """Return the run's input mode from the header, or an empty string."""
    for line in text.splitlines():
        if line.startswith(MODE_PREFIX):
            return line[len(MODE_PREFIX):].strip()
        if line.startswith("==="):
            break
    return ""


def run_files(directory):
    """Return the saved run files in a directory, in a stable order."""
    if not os.path.isdir(directory):
        return []
    return sorted(
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if name.lower().endswith(".txt")
    )


def revalidate_file(path):
    """Revalidate one saved run and return a CSV row, or None if unreadable."""
    try:
        text = read_text(path)
    except OSError as error:
        print(f"Skipping {os.path.basename(path)}: {error}")
        return None
    quiz = extract_quiz(text)
    if quiz is None:
        print(f"Skipping {os.path.basename(path)}: no "
              f"'{QUIZ_MARKER}' section found.")
        return None
    result = validate_quiz(quiz)
    return {
        "File": os.path.basename(path),
        "Mode": extract_mode(text),
        "Questions found": result.questions_found,
        "Returned five": result.returned_five,
        "Validation passed": result.passed,
        "Missing labels": result.count_faults(MISSING_FAULT),
        "Empty values": result.count_faults(EMPTY_FAULT),
        "Placeholder echoes": result.count_faults(PLACEHOLDER_FAULT),
        "Faults": result.describe(),
        "Invalid source basis": result.count_faults(INVALID_BASIS_FAULT),
        "Generated notes basis": result.count_faults(GENERATED_NOTES_FAULT),
        "Shape warnings": result.shape_warning_count,
        # Warnings are reported alongside the faults, never inside them, so
        # the pass or fail verdict stays decided by the faults alone.
        "Warnings": result.describe_warnings(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", default="outputs",
                        help="Directory of saved run files.")
    parser.add_argument("--csv", default=None,
                        help="CSV to write (default: "
                             "<outputs>/revalidated_quizzes.csv).")
    args = parser.parse_args(argv)

    paths = run_files(args.outputs)
    if not paths:
        print(f"No saved run files found in '{args.outputs}'.")
        return 1

    rows = [row for row in (revalidate_file(path) for path in paths)
            if row is not None]
    if not rows:
        print("No saved run contained a quiz section to revalidate.")
        return 1

    for row in rows:
        verdict = "passed" if row["Validation passed"] else "FAILED"
        print(f"{row['File']}")
        print(f"  questions found : {row['Questions found']}"
              f"   returned five: {row['Returned five']}")
        print(f"  validation      : {verdict}")
        print(f"  faults          : {row['Faults']}")

    passed = sum(1 for row in rows if row["Validation passed"])
    returned_five = sum(1 for row in rows if row["Returned five"])
    total = len(rows)
    print()
    print(f"{passed} of {total} runs passed quiz validation "
          f"({passed / total:.0%}).")
    print(f"{returned_five} of {total} runs returned exactly five questions "
          f"({returned_five / total:.0%}).")

    csv_path = args.csv or os.path.join(args.outputs,
                                        "revalidated_quizzes.csv")
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Written to {csv_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
