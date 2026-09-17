"""Tests for the extractor comparison harness.

Cover the two things the report relies on: that unavailable paths are skipped
rather than crashing, and that the CSV columns match the OCR 5.2 sheet in the
exact order and spelling. Path availability is patched so the tests are
deterministic no matter what is installed on the machine, and no test makes a
real model call.
"""
import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

from evaluation import compare_extractors as ce
from tests.fakes import FakeLLMClient


class CsvColumnTests(unittest.TestCase):
    def test_manual_columns_exact_order_and_spelling(self):
        self.assertEqual(ce.MANUAL_COLUMNS, [
            "Slide set",
            "Pages",
            "Extraction path",
            "Missing text (1 to 5)",
            "Incorrect text (1 to 5)",
            "Structure kept (1 to 5)",
            "Overall (1 to 5)",
            "Notes",
        ])

    def test_auto_columns_follow_the_manual_columns(self):
        n = len(ce.MANUAL_COLUMNS)
        self.assertEqual(ce.CSV_COLUMNS[:n], ce.MANUAL_COLUMNS)
        self.assertEqual(ce.CSV_COLUMNS[n:], ce.AUTO_COLUMNS)


class BuildPathsTests(unittest.TestCase):
    def test_pdf_all_available_runs_three_paths_in_order(self):
        runnable, skipped = ce.build_paths(".pdf", True, True, True)
        self.assertEqual(runnable,
                         [ce.PYPDF_PATH, ce.TESSERACT_PATH, ce.VISION_PATH])
        # The native pptx path is reported as skipped rather than omitted, so
        # the run states the full matrix of techniques considered.
        self.assertEqual({name for name, _ in skipped},
                         {ce.NATIVE_PPTX_PATH})

    def test_all_unavailable_skips_every_path(self):
        runnable, skipped = ce.build_paths(".pdf", False, False, False)
        self.assertEqual(runnable, [])
        self.assertEqual({name for name, _ in skipped},
                         {ce.NATIVE_PPTX_PATH, ce.PYPDF_PATH,
                          ce.TESSERACT_PATH, ce.VISION_PATH})

    def test_image_input_skips_pypdf(self):
        runnable, skipped = ce.build_paths(".png", True, True, True)
        self.assertNotIn(ce.PYPDF_PATH, runnable)
        self.assertIn(ce.TESSERACT_PATH, runnable)
        self.assertIn(ce.VISION_PATH, runnable)
        self.assertIn(ce.PYPDF_PATH, [name for name, _ in skipped])

    def test_pdf_without_rendering_skips_the_image_paths(self):
        runnable, skipped = ce.build_paths(".pdf", True, True, False)
        self.assertEqual(runnable, [ce.PYPDF_PATH])
        skipped_names = {name for name, _ in skipped}
        self.assertIn(ce.TESSERACT_PATH, skipped_names)
        self.assertIn(ce.VISION_PATH, skipped_names)


class PptxPathTests(unittest.TestCase):
    """A pptx yields four comparable techniques, but only once LibreOffice can
    turn it into pixels; without it the native text path must still run."""

    def test_pptx_with_everything_available_runs_all_four_paths(self):
        runnable, skipped = ce.build_paths(
            ".pptx", True, True, True, pptx_ok=True, libreoffice_ok=True)
        self.assertEqual(runnable, [ce.NATIVE_PPTX_PATH, ce.PYPDF_PATH,
                                    ce.TESSERACT_PATH, ce.VISION_PATH])
        self.assertEqual(skipped, [])

    def test_pptx_without_libreoffice_runs_only_the_native_path(self):
        runnable, skipped = ce.build_paths(
            ".pptx", True, True, True, pptx_ok=True, libreoffice_ok=False)
        self.assertEqual(runnable, [ce.NATIVE_PPTX_PATH])
        skipped_names = {name for name, _ in skipped}
        self.assertEqual(skipped_names,
                         {ce.PYPDF_PATH, ce.TESSERACT_PATH, ce.VISION_PATH})

    def test_pptx_without_python_pptx_skips_the_native_path(self):
        runnable, _skipped = ce.build_paths(
            ".pptx", True, True, True, pptx_ok=False, libreoffice_ok=True)
        self.assertNotIn(ce.NATIVE_PPTX_PATH, runnable)

    def test_native_path_is_not_offered_for_a_pdf(self):
        runnable, _skipped = ce.build_paths(
            ".pdf", True, True, True, pptx_ok=True, libreoffice_ok=True)
        self.assertNotIn(ce.NATIVE_PPTX_PATH, runnable)

    def test_pptx_with_nothing_available_reports_and_exits(self):
        with mock.patch.object(ce.slides_ocr, "pdf_available",
                               return_value=False), \
                mock.patch.object(ce.slides_ocr, "ocr_available",
                                  return_value=False), \
                mock.patch.object(ce.slide_vision, "is_available",
                                  return_value=False), \
                mock.patch.object(ce.slides_pptx, "is_available",
                                  return_value=False), \
                mock.patch.object(ce.slides_pptx, "libreoffice_available",
                                  return_value=False):
            buffer = io.StringIO()
            with tempfile.TemporaryDirectory() as out, redirect_stdout(buffer):
                code = ce.main(["--input", "deck.pptx", "--out", out],
                               client=FakeLLMClient())
        self.assertEqual(code, 1)
        self.assertIn("LibreOffice was not found", buffer.getvalue())


class MainSkipTests(unittest.TestCase):
    def test_main_returns_1_and_writes_no_csv_when_nothing_available(self):
        with mock.patch.object(ce.slides_ocr, "pdf_available",
                               return_value=False), \
                mock.patch.object(ce.slides_ocr, "ocr_available",
                                  return_value=False), \
                mock.patch.object(ce.slide_vision, "is_available",
                                  return_value=False):
            buffer = io.StringIO()
            with tempfile.TemporaryDirectory() as out, redirect_stdout(buffer):
                code = ce.main(["--input", "deck.pdf", "--out", out],
                               client=FakeLLMClient())
                csv_exists = os.path.exists(
                    os.path.join(out, "extractor_comparison.csv"))
        self.assertEqual(code, 1)
        self.assertIn("No extraction paths", buffer.getvalue())
        self.assertFalse(csv_exists)


class VisionPreflightTests(unittest.TestCase):
    def test_unreachable_vision_model_is_skipped_before_rendering(self):
        """An unpulled model must be reported up front and must not trigger
        page rendering, which is the slow work the pre-flight exists to
        avoid."""
        render = mock.patch.object(ce.slide_vision, "render_pdf_to_images")
        with mock.patch.object(ce.slides_ocr, "pdf_available",
                               return_value=False), \
                mock.patch.object(ce.slides_ocr, "ocr_available",
                                  return_value=False), \
                mock.patch.object(ce.slide_vision, "is_available",
                                  return_value=True), \
                render as render_mock:
            buffer = io.StringIO()
            with tempfile.TemporaryDirectory() as out, redirect_stdout(buffer):
                code = ce.main(["--input", "deck.pdf", "--out", out],
                               client=FakeLLMClient(model="qwen3-vl:4b",
                                                    reachable=False))
        self.assertEqual(code, 1)
        self.assertIn("is not available", buffer.getvalue())
        render_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
