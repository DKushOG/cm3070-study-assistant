"""Tests for the batch evaluation runner.

Part of the full pipeline integration group.
"""

import csv
import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

from evaluation import run_eval
from evaluation.rubric import CRITERIA
from tests.fakes import FakeLLMClient


def write(path, text):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


class RunEvalTests(unittest.TestCase):
    """The whole batch, from sample files to a written CSV.

        A mode whose sources are missing is skipped with a warning rather than
        ending the run.

    """

    def run_main(self, samples, out, client):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = run_eval.main(["--samples", samples, "--out", out],
                                 client=client)
        return code, buffer.getvalue()

    def test_full_batch_runs_four_modes_and_writes_csv(self):
        client = FakeLLMClient()
        with tempfile.TemporaryDirectory() as samples, \
                tempfile.TemporaryDirectory() as out, \
                mock.patch.object(run_eval.config, "QUIZ_MAX_ATTEMPTS", 1):
            # Pinned rather than read from the ambient .env: the call count is
            # attempts-dependent, so leaving it to the developer's environment
            # made this assertion pass or fail according to a setting the test
            # is not about.
            write(os.path.join(samples, "transcript_sample.txt"), "t")
            write(os.path.join(samples, "slide_text_sample.txt"), "s")
            write(os.path.join(samples, "notes_sample.txt"), "n")
            code, _output = self.run_main(samples, out, client)
            self.assertEqual(code, 0)
            self.assertEqual(len(client.prompts), 8)  # 2 calls x 4 modes
            csv_path = os.path.join(out, "evaluation_scores.csv")
            with open(csv_path, newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 4)
            self.assertEqual(rows[0]["mode"], "Transcript only")
            self.assertEqual(rows[3]["mode"],
                             "Transcript, slide text and notes")
            for key, _description in CRITERIA:
                self.assertIn(key, rows[0])
                self.assertEqual(rows[0][key], "")
            saved = [name for name in os.listdir(out)
                     if name.endswith(".txt")]
            self.assertEqual(len(saved), 4)

    def test_partial_samples_skip_unrunnable_modes(self):
        client = FakeLLMClient()
        with tempfile.TemporaryDirectory() as samples, \
                tempfile.TemporaryDirectory() as out:
            write(os.path.join(samples, "transcript_sample.txt"), "t")
            code, output = self.run_main(samples, out, client)
            self.assertEqual(code, 0)
            self.assertIn("Skipping", output)
            csv_path = os.path.join(out, "evaluation_scores.csv")
            with open(csv_path, newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["mode"], "Transcript only")

    def test_no_samples_reports_error_without_crashing(self):
        client = FakeLLMClient()
        with tempfile.TemporaryDirectory() as samples, \
                tempfile.TemporaryDirectory() as out:
            code, output = self.run_main(samples, out, client)
            self.assertEqual(code, 1)
            self.assertIn("No input modes can run", output)
            self.assertEqual(len(client.prompts), 0)
            self.assertFalse(os.path.exists(
                os.path.join(out, "evaluation_scores.csv")))


class RepeatsAndColumnsTests(unittest.TestCase):
    """Repeats and the columns that make a results file self-describing.

        The scoring columns stay blank, since they are filled in by hand.

    """

    def run_main(self, argv, client):
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            code = run_eval.main(argv, client=client)
        return code, buffer.getvalue()

    def write_all_samples(self, samples):
        write(os.path.join(samples, "transcript_sample.txt"), "t")
        write(os.path.join(samples, "slide_text_sample.txt"), "s")
        write(os.path.join(samples, "notes_sample.txt"), "n")

    def test_repeats_produce_a_row_per_run_with_a_run_column(self):
        client = FakeLLMClient()
        with tempfile.TemporaryDirectory() as samples, \
                tempfile.TemporaryDirectory() as out:
            self.write_all_samples(samples)
            code, _ = self.run_main(
                ["--samples", samples, "--out", out, "--repeats", "3"], client)
            self.assertEqual(code, 0)
            with open(os.path.join(out, "evaluation_scores.csv"),
                      newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        # Four modes x three runs, every output saved separately.
        self.assertEqual(len(rows), 12)
        self.assertEqual({row["run"] for row in rows}, {"1", "2", "3"})

    def test_default_repeats_is_one(self):
        client = FakeLLMClient()
        with tempfile.TemporaryDirectory() as samples, \
                tempfile.TemporaryDirectory() as out:
            self.write_all_samples(samples)
            self.run_main(["--samples", samples, "--out", out], client)
            with open(os.path.join(out, "evaluation_scores.csv"),
                      newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 4)
        self.assertEqual({row["run"] for row in rows}, {"1"})

    def test_metadata_and_validation_columns_are_present(self):
        client = FakeLLMClient(model="llama3.2")
        quiz = "\n".join(
            f"Question: Q{n}?\nSuggested answer: A{n}.\n"
            f"Question type: definition\nSource basis: transcript\n"
            for n in range(1, 6))
        client._replies = ["NOTES", quiz]
        with tempfile.TemporaryDirectory() as samples, \
                tempfile.TemporaryDirectory() as out:
            self.write_all_samples(samples)
            self.run_main(
                ["--samples", samples, "--out", out,
                 "--slide-source", "Tesseract OCR"], client)
            with open(os.path.join(out, "evaluation_scores.csv"),
                      newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        row = rows[0]
        for column in ("run", "model", "temperature", "seed", "slide_source",
                       "questions_found", "returned_five", "attempts",
                       "validation_passed"):
            self.assertIn(column, row)
        self.assertEqual(row["model"], "llama3.2")
        self.assertEqual(row["slide_source"], "Tesseract OCR")
        self.assertEqual(row["attempts"], "1")

    def test_unset_temperature_and_seed_are_blank_not_none(self):
        client = FakeLLMClient()
        with tempfile.TemporaryDirectory() as samples, \
                tempfile.TemporaryDirectory() as out:
            self.write_all_samples(samples)
            self.run_main(["--samples", samples, "--out", out], client)
            with open(os.path.join(out, "evaluation_scores.csv"),
                      newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(rows[0]["temperature"], "")
        self.assertEqual(rows[0]["seed"], "")

    def test_new_columns_are_appended_after_the_rubric_columns(self):
        """Appended last so no existing column, rubric ones included, moves."""
        client = FakeLLMClient()
        with tempfile.TemporaryDirectory() as samples, \
                tempfile.TemporaryDirectory() as out:
            self.write_all_samples(samples)
            self.run_main(["--samples", samples, "--out", out], client)
            with open(os.path.join(out, "evaluation_scores.csv"),
                      newline="", encoding="utf-8") as handle:
                columns = next(csv.reader(handle))
        self.assertEqual(columns[-5:], [
            "invalid_source_basis", "generated_notes_basis",
            "shape_warnings", "quiz_seeds", "quiz_structured",
        ])
        rubric_names = [key for key, _description in CRITERIA]
        self.assertEqual(columns[-5 - len(rubric_names):-5], rubric_names)

    def test_rubric_columns_stay_blank_for_manual_scoring(self):
        client = FakeLLMClient()
        with tempfile.TemporaryDirectory() as samples, \
                tempfile.TemporaryDirectory() as out:
            self.write_all_samples(samples)
            self.run_main(["--samples", samples, "--out", out], client)
            with open(os.path.join(out, "evaluation_scores.csv"),
                      newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        for key, _description in CRITERIA:
            self.assertEqual(rows[0][key], "")


class BuildModesTests(unittest.TestCase):
    """Which modes are built from the sample files that exist."""

    def test_all_sources_gives_four_modes(self):
        runnable, skipped = run_eval.build_modes("t", "s", "n")
        self.assertEqual(len(runnable), 4)
        self.assertEqual(skipped, [])

    def test_whitespace_source_counts_as_missing(self):
        runnable, skipped = run_eval.build_modes("t", "   ", None)
        self.assertEqual([name for name, _ in runnable],
                         ["Transcript only"])
        self.assertEqual(len(skipped), 3)


if __name__ == "__main__":
    unittest.main()
