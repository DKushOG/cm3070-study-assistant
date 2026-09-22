"""Tests for choosing and running a slide extraction method.

The interface calls modules/slide_extraction.py, so these tests are what
stand behind the claim that the application offers per-slide routing by
default, reads the whole deck and reports slides the vision model could not
read. Every optional dependency and every model call is replaced, so no
test here needs PyMuPDF, LibreOffice, python-pptx or a running model.
"""
import unittest
from unittest import mock

import config
from modules import slide_extraction as se


class _Available:
    """Context manager making every method available, with real labels."""

    def __init__(self, libreoffice=True, rendering=True):
        self.libreoffice = libreoffice
        self.rendering = rendering

    def __enter__(self):
        self.patchers = [
            mock.patch("modules.slides_pptx.libreoffice_available",
                       return_value=self.libreoffice),
            mock.patch("modules.slides_pptx.is_available", return_value=True),
            mock.patch("modules.slides_ocr.pdf_available", return_value=True),
            mock.patch("modules.slides_ocr.ocr_available", return_value=True),
            mock.patch("modules.slide_vision.is_available",
                       return_value=self.rendering),
        ]
        for patcher in self.patchers:
            patcher.start()
        return self

    def __exit__(self, *exc):
        for patcher in self.patchers:
            patcher.stop()
        return False


class OfferedMethodTests(unittest.TestCase):
    def test_routing_is_offered_first_for_a_powerpoint_deck(self):
        with _Available():
            paths = se.paths_for(".pptx")
        self.assertEqual(paths[0], se.routed_path_label())
        self.assertIn(se.vision_path_label(), paths)
        self.assertIn(se.NATIVE_PPTX_PATH, paths)

    def test_routing_is_offered_first_for_a_pdf(self):
        with _Available():
            paths = se.paths_for(".pdf")
        self.assertEqual(paths[0], se.routed_path_label())
        self.assertNotIn(se.NATIVE_PPTX_PATH, paths)

    def test_routing_is_not_offered_for_a_single_image(self):
        """One image has nothing to route across."""
        with _Available():
            paths = se.paths_for(".png")
        self.assertNotIn(se.routed_path_label(), paths)
        self.assertIn(se.vision_path_label(), paths)

    def test_without_libreoffice_a_deck_offers_native_text_only(self):
        with _Available(libreoffice=False):
            paths = se.paths_for(".pptx")
        self.assertEqual(paths, [se.NATIVE_PPTX_PATH])

    def test_without_page_rendering_neither_model_method_is_offered(self):
        with _Available(rendering=False):
            paths = se.paths_for(".pdf")
        self.assertNotIn(se.routed_path_label(), paths)
        self.assertNotIn(se.vision_path_label(), paths)

    def test_labels_name_the_configured_vision_model(self):
        self.assertIn(config.VISION_MODEL, se.routed_path_label())
        self.assertIn(config.VISION_MODEL, se.vision_path_label())

    def test_only_the_two_model_methods_need_the_vision_model(self):
        self.assertTrue(se.needs_vision_model(se.routed_path_label()))
        self.assertTrue(se.needs_vision_model(se.vision_path_label()))
        self.assertFalse(se.needs_vision_model(se.NATIVE_PPTX_PATH))
        self.assertFalse(se.needs_vision_model(se.CLASSICAL_PATH))


class DefaultsTests(unittest.TestCase):
    def test_vision_model_default_is_the_selected_model(self):
        """The rejected model must not come back as the default."""
        self.assertEqual(config.VISION_MODEL, "qwen2.5vl:7b")

    def test_page_cap_reads_a_whole_lecture_deck(self):
        """The old cap of 20 silently dropped the end of longer decks."""
        self.assertGreaterEqual(config.VISION_MAX_PAGES, 100)

    def test_interface_uses_structured_quiz_unless_switched_off(self):
        self.assertTrue(config.QUIZ_STRUCTURED_IN_APP)

    def test_batch_runner_keeps_the_legacy_reference_default(self):
        self.assertFalse(config.QUIZ_STRUCTURED)


class ExtractDispatchTests(unittest.TestCase):
    def test_routing_passes_the_page_cap_and_reports_its_summary(self):
        report = {"pages": 10, "vision_pages": 4, "text_pages": 6,
                  "unreadable_pages": [], "truncated_pages": []}
        with mock.patch("modules.slide_routing.extract_routed",
                        return_value=("TEXT", report)) as routed:
            text, notice = se.extract("deck.pptx", se.routed_path_label(),
                                      vision_client=object())
        self.assertEqual(text, "TEXT")
        self.assertEqual(routed.call_args.kwargs["max_pages"],
                         config.VISION_MAX_PAGES)
        self.assertIn("4 of 10 slides sent to the vision model", notice)

    def test_routing_notice_names_unreadable_slides(self):
        report = {"pages": 5, "vision_pages": 3, "text_pages": 2,
                  "unreadable_pages": [4], "truncated_pages": []}
        with mock.patch("modules.slide_routing.extract_routed",
                        return_value=("TEXT", report)):
            _text, notice = se.extract("deck.pdf", se.routed_path_label(),
                                       vision_client=object())
        self.assertIn("could not be read", notice)
        self.assertIn("4", notice)

    def test_vision_on_a_pdf_renders_up_to_the_configured_cap(self):
        """The earlier interface called the PDF path with the module's own
        default of 20 pages, ignoring the configured cap."""
        with mock.patch("modules.slide_vision.render_pdf_to_images",
                        return_value=["img"]) as render, \
                mock.patch("modules.slide_vision.extract_from_images_with_report",
                           return_value=("T", {"pages": 1,
                                               "unreadable_pages": [],
                                               "truncated_pages": []})):
            text, notice = se.extract("deck.pdf", se.vision_path_label(),
                                      vision_client=object())
        self.assertEqual(render.call_args.kwargs["max_pages"],
                         config.VISION_MAX_PAGES)
        self.assertEqual((text, notice), ("T", ""))

    def test_vision_on_a_deck_uses_the_reporting_extractor(self):
        report = {"pages": 2, "unreadable_pages": [2], "truncated_pages": []}
        with mock.patch("modules.slides_pptx.render_pptx_to_images",
                        return_value=["a", "b"]), \
                mock.patch("modules.slide_vision.extract_from_images_with_report",
                           return_value=("T", report)):
            _text, notice = se.extract("deck.pptx", se.vision_path_label(),
                                       vision_client=object())
        self.assertIn("1 of 2 slides could not be read", notice)

    def test_native_text_never_includes_speaker_notes(self):
        with mock.patch("modules.slides_pptx.extract_pptx_text",
                        return_value="N") as native:
            text, notice = se.extract("deck.pptx", se.NATIVE_PPTX_PATH)
        self.assertEqual((text, notice), ("N", ""))
        self.assertFalse(native.call_args.kwargs["include_notes"])

    def test_unknown_method_is_rejected_clearly(self):
        with self.assertRaises(ValueError):
            se.extract("deck.pdf", "Something else")


if __name__ == "__main__":
    unittest.main()
