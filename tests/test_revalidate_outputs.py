"""Tests for revalidating already-saved runs.

Files are constructed in a temporary directory rather than read from the real
outputs/ folder, so the suite does not depend on which runs happen to be on the
machine. Windows line endings and the metadata header are exercised
deliberately, since those are what the parser has to survive.
"""
import csv
import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout

from evaluation import revalidate_outputs as ro

GOOD_QUIZ = "\n".join(
    f"Question: Q{n}?\nSuggested answer: A{n}.\n"
    f"Question type: definition\nSource basis: transcript\n"
    for n in range(1, 6))

HEADER = (
    "MODE: Combined transcript and slide text\n"
    "model=llama3.2\n"
    "base_url=http://localhost:11434/v1\n"
    "temperature=unset\n"
    "seed=unset\n"
    "quiz_attempts=1\n"
    "truncation: none\n\n"
)


def write_run(directory, name, quiz, line_ending="\r\n"):
    """Write a saved run in the real file layout, CRLF by default."""
    text = (HEADER
            + "=== REVISION NOTES ===\nSome notes.\n\n"
            + "=== QUIZ QUESTIONS ===\n" + quiz + "\n\n"
            + "notes_seconds=1.00\nquiz_seconds=2.00\ntotal_seconds=3.00\n")
    path = os.path.join(directory, name)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(text.replace("\n", line_ending))
    return path


class ExtractQuizTests(unittest.TestCase):
    def test_windows_line_endings_are_normalised(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_run(tmp, "run.txt", GOOD_QUIZ)
            text = ro.read_text(path)
        self.assertNotIn("\r", text)

    def test_quiz_section_excludes_header_and_timing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_run(tmp, "run.txt", GOOD_QUIZ)
            quiz = ro.extract_quiz(ro.read_text(path))
        self.assertIn("Question: Q1?", quiz)
        self.assertNotIn("MODE:", quiz)
        self.assertNotIn("REVISION NOTES", quiz)
        self.assertNotIn("notes_seconds", quiz)
        self.assertNotIn("total_seconds", quiz)

    def test_missing_marker_returns_none_not_empty_string(self):
        """None means 'no quiz section', which is a different fact from an
        empty quiz section."""
        self.assertIsNone(ro.extract_quiz("MODE: x\nnotes_seconds=1.00\n"))

    def test_mode_is_read_from_the_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_run(tmp, "run.txt", GOOD_QUIZ)
            mode = ro.extract_mode(ro.read_text(path))
        self.assertEqual(mode, "Combined transcript and slide text")


class RevalidateFileTests(unittest.TestCase):
    def test_valid_quiz_reports_passed(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_run(tmp, "run.txt", GOOD_QUIZ)
            row = ro.revalidate_file(path)
        self.assertEqual(row["Questions found"], 5)
        self.assertTrue(row["Returned five"])
        self.assertTrue(row["Validation passed"])

    def test_file_without_a_quiz_section_is_skipped_not_fatal(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "truncated.txt")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("MODE: x\nmodel=llama3.2\n")
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                row = ro.revalidate_file(path)
        self.assertIsNone(row)
        self.assertIn("Skipping", buffer.getvalue())


class MainTests(unittest.TestCase):
    def run_main(self, argv):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = ro.main(argv)
        return code, buffer.getvalue()

    def test_batch_writes_csv_and_reports_the_pass_proportion(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_run(tmp, "a_good.txt", GOOD_QUIZ)
            write_run(tmp, "b_bad.txt", "Question: only one?\n")
            code, output = self.run_main(["--outputs", tmp])
            csv_path = os.path.join(tmp, "revalidated_quizzes.csv")
            with open(csv_path, newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(code, 0)
        self.assertEqual(len(rows), 2)
        self.assertEqual(list(rows[0].keys()), ro.CSV_COLUMNS)
        self.assertIn("1 of 2 runs passed quiz validation (50%)", output)

    def test_new_columns_are_appended_after_the_existing_ones(self):
        """The evaluation workbook reads these by position as well as name, so
        existing columns must not move."""
        self.assertEqual(ro.CSV_COLUMNS[:9], [
            "File", "Mode", "Questions found", "Returned five",
            "Validation passed", "Missing labels", "Empty values",
            "Placeholder echoes", "Faults",
        ])
        self.assertEqual(ro.CSV_COLUMNS[9:], [
            "Invalid source basis", "Generated notes basis",
            "Shape warnings", "Warnings",
        ])

    def test_source_basis_and_warning_columns_are_populated(self):
        quiz = "\n\n".join(
            f"Question: Statement {n} is a fact.\nSuggested answer: A{n}.\n"
            f"Question type: definition\nSource basis: Revision notes"
            for n in range(1, 6))
        with tempfile.TemporaryDirectory() as tmp:
            path = write_run(tmp, "run.txt", quiz)
            row = ro.revalidate_file(path)
        self.assertEqual(row["Generated notes basis"], 5)
        self.assertEqual(row["Shape warnings"], 5)
        self.assertIn("does not read as a question", row["Warnings"])
        # Warnings are reported beside the faults, never inside them.
        self.assertNotIn("does not read as a question", row["Faults"])

    def test_empty_directory_reports_without_crashing(self):
        with tempfile.TemporaryDirectory() as tmp:
            code, output = self.run_main(["--outputs", tmp])
        self.assertEqual(code, 1)
        self.assertIn("No saved run files", output)

    def test_missing_directory_reports_without_crashing(self):
        code, output = self.run_main(["--outputs", "no_such_directory_here"])
        self.assertEqual(code, 1)
        self.assertIn("No saved run files", output)

    def test_csv_path_can_be_overridden(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_run(tmp, "a_good.txt", GOOD_QUIZ)
            target = os.path.join(tmp, "custom.csv")
            code, _output = self.run_main(
                ["--outputs", tmp, "--csv", target])
            self.assertEqual(code, 0)
            self.assertTrue(os.path.exists(target))


if __name__ == "__main__":
    unittest.main()
