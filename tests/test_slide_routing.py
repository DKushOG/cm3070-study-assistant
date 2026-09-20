"""Tests for the per-slide extraction routing rule.

The rule decides, before any model call is made, whether a slide is worth
sending to the vision model. These tests fix the decision boundaries and the
asymmetry the rule is built on, so neither can be changed by accident later
without a test saying so.

No test here makes a model call or reads a real deck.
"""
import unittest

from modules import slide_routing


class ThresholdTests(unittest.TestCase):
    """The shipped thresholds must match the ones that were measured.

    Written because a threshold is a number in a file, and a number in a file
    drifts. The values below are the ones derived from 111 instrumented
    slides, and a change to either should be a deliberate act with new
    evidence behind it rather than a tidy-up.
    """

    def test_picture_area_threshold_is_the_measured_one(self):
        self.assertAlmostEqual(slide_routing.PICTURE_AREA_THRESHOLD, 0.10)

    def test_text_fallback_threshold_is_the_measured_one(self):
        self.assertEqual(slide_routing.TEXT_WORDS_THRESHOLD, 60)


class PictureAreaRuleTests(unittest.TestCase):
    def test_a_picture_heavy_slide_goes_to_the_vision_model(self):
        route, reason = slide_routing.decide(picture_area_ratio=0.74)
        self.assertEqual(route, slide_routing.ROUTE_VISION)
        self.assertIn("74%", reason)

    def test_a_text_slide_is_read_from_the_text_layer(self):
        route, _reason = slide_routing.decide(picture_area_ratio=0.02)
        self.assertEqual(route, slide_routing.ROUTE_TEXT)

    def test_the_boundary_is_inclusive(self):
        # At exactly the threshold the slide is sent. Recall is preferred at
        # the margin because a slide wrongly skipped loses its diagram
        # permanently, while a slide wrongly sent costs a few seconds.
        route, _reason = slide_routing.decide(
            picture_area_ratio=slide_routing.PICTURE_AREA_THRESHOLD)
        self.assertEqual(route, slide_routing.ROUTE_VISION)

    def test_picture_area_wins_over_text_yield_when_both_are_known(self):
        """A dense slide that is mostly picture still goes to vision.

        Picture area correlates with the outcome at +0.674 and text yield at
        -0.485, so where they disagree the stronger predictor decides.
        """
        route, _reason = slide_routing.decide(picture_area_ratio=0.60,
                                              text_layer_words=190)
        self.assertEqual(route, slide_routing.ROUTE_VISION)


class TextYieldFallbackTests(unittest.TestCase):
    def test_a_sparse_pdf_page_goes_to_the_vision_model(self):
        route, reason = slide_routing.decide(text_layer_words=12)
        self.assertEqual(route, slide_routing.ROUTE_VISION)
        self.assertIn("12 words", reason)

    def test_a_wordy_pdf_page_is_read_from_the_text_layer(self):
        route, _reason = slide_routing.decide(text_layer_words=190)
        self.assertEqual(route, slide_routing.ROUTE_TEXT)

    def test_the_fallback_boundary(self):
        below, _ = slide_routing.decide(
            text_layer_words=slide_routing.TEXT_WORDS_THRESHOLD - 1)
        at, _ = slide_routing.decide(
            text_layer_words=slide_routing.TEXT_WORDS_THRESHOLD)
        self.assertEqual(below, slide_routing.ROUTE_VISION)
        self.assertEqual(at, slide_routing.ROUTE_TEXT)


class NoFeaturesTests(unittest.TestCase):
    def test_with_nothing_to_go_on_the_slide_is_sent(self):
        """Defaulting to vision is the deliberate choice, not an oversight.

        The two errors are not symmetric: skipping a slide that needed the
        model loses content silently and permanently, and calling one that
        did not costs about eight seconds.
        """
        route, reason = slide_routing.decide()
        self.assertEqual(route, slide_routing.ROUTE_VISION)
        self.assertIn("defaulting", reason)


class CustomThresholdTests(unittest.TestCase):
    def test_thresholds_can_be_overridden_for_an_ablation(self):
        route, _reason = slide_routing.decide(picture_area_ratio=0.30,
                                              picture_threshold=0.50)
        self.assertEqual(route, slide_routing.ROUTE_TEXT)


class PictureAreaReadingTests(unittest.TestCase):
    def test_a_non_pptx_yields_no_ratios(self):
        # Which is what makes a PDF fall back to text yield rather than
        # silently routing everything one way.
        self.assertEqual(slide_routing.pptx_picture_area_ratios("deck.pdf"),
                         {})

    def test_a_missing_file_does_not_raise(self):
        self.assertEqual(
            slide_routing.pptx_picture_area_ratios("no_such_deck.pptx"), {})


class SummaryTests(unittest.TestCase):
    def test_a_routed_run_describes_what_it_avoided(self):
        summary = slide_routing.summarise(
            {"pages": 100, "vision_pages": 57})
        self.assertIn("57 of 100", summary)
        self.assertIn("43%", summary)

    def test_an_empty_report_says_nothing(self):
        self.assertEqual(slide_routing.summarise({}), "")
        self.assertEqual(slide_routing.summarise(None), "")


if __name__ == "__main__":
    unittest.main()
