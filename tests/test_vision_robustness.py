"""Tests for detecting and reporting slides the vision model cannot read.

These exist because of a fault that survived every previous test and every
previous measurement. When the vision model ran out of room it returned an
empty string and raised nothing, and the extraction path joined that empty
string into the output exactly as it would join a blank slide. Forty-nine of
111 slides in the instrumentation corpus, and 15 of 36 pages in the extractor
comparison behind the draft report, were recorded as successfully extracted
when nothing had come back at all.

The lesson is narrower than "add a test". A silent failure is invisible to
any measurement taken over it, so the fix has to be detection at the point of
the call, and the tests below fix the detection in place so it cannot be
removed by accident later.

No test here makes a real model call.
"""
import unittest

from modules import slide_vision
from tests.fakes import FakeLLMClient

IMAGES = ["data:image/png;base64,AAA", "data:image/png;base64,BBB"]


class EmptyReplyDetectionTests(unittest.TestCase):
    def test_empty_reply_is_reported_not_silently_joined(self):
        # Every call returns nothing, which is what the model actually did.
        client = FakeLLMClient(replies=[""])
        text, report = slide_vision.extract_from_images_with_report(
            IMAGES, client, max_retries=0)
        self.assertEqual(report["unreadable_pages"], [1, 2])
        self.assertEqual(report["pages"], 2)
        # The reader sees a note rather than a page marker with nothing under
        # it, so a missing slide is visible in the output itself.
        self.assertIn(slide_vision.UNREADABLE_NOTE, text)

    def test_a_readable_deck_reports_nothing(self):
        client = FakeLLMClient(replies=["PAGE ONE", "PAGE TWO"])
        text, report = slide_vision.extract_from_images_with_report(
            IMAGES, client)
        self.assertEqual(report["unreadable_pages"], [])
        self.assertEqual(report["truncated_pages"], [])
        self.assertEqual(report["retries"], 0)
        self.assertIn("PAGE ONE", text)
        self.assertIn("PAGE TWO", text)
        self.assertNotIn(slide_vision.UNREADABLE_NOTE, text)

    def test_whitespace_only_reply_counts_as_unreadable(self):
        # A reply of spaces and newlines is not content, and the model does
        # sometimes return one.
        client = FakeLLMClient(replies=["   \n  "])
        _text, report = slide_vision.extract_from_images_with_report(
            ["data:image/png;base64,AAA"], client, max_retries=0)
        self.assertEqual(report["unreadable_pages"], [1])


class RetryTests(unittest.TestCase):
    def test_an_empty_page_is_retried_and_can_recover(self):
        # First call empty, second call good. One image, so the recovery must
        # come from the retry rather than from moving on to another page.
        client = FakeLLMClient(replies=["", "RECOVERED TEXT"])
        text, report = slide_vision.extract_from_images_with_report(
            ["data:image/png;base64,AAA"], client, max_retries=1)
        self.assertEqual(len(client.image_calls), 2)
        self.assertEqual(report["retries"], 1)
        self.assertEqual(report["unreadable_pages"], [])
        self.assertIn("RECOVERED TEXT", text)

    def test_a_page_that_answers_is_not_retried(self):
        client = FakeLLMClient(replies=["GOOD TEXT"])
        _text, report = slide_vision.extract_from_images_with_report(
            ["data:image/png;base64,AAA"], client, max_retries=1)
        self.assertEqual(len(client.image_calls), 1)
        self.assertEqual(report["retries"], 0)

    def test_retries_can_be_disabled(self):
        client = FakeLLMClient(replies=[""])
        _text, report = slide_vision.extract_from_images_with_report(
            ["data:image/png;base64,AAA"], client, max_retries=0)
        self.assertEqual(len(client.image_calls), 1)
        self.assertEqual(report["retries"], 0)


class TruncationTests(unittest.TestCase):
    def test_a_truncated_page_is_recorded_even_though_it_has_text(self):
        # This is the case that mattered most: the page produced words, so
        # every length-based check passed, but it stopped in mid-sentence
        # because it ran out of room. Text alone cannot reveal that; only the
        # finish reason can.
        client = FakeLLMClient(replies=["HALF A SENTENCE WHICH STOPS"],
                               finish_reasons=["length"])
        _text, report = slide_vision.extract_from_images_with_report(
            ["data:image/png;base64,AAA"], client, max_retries=0)
        self.assertEqual(report["truncated_pages"], [1])
        # It is truncated, not unreadable. The distinction matters, because
        # partial content is still worth keeping.
        self.assertEqual(report["unreadable_pages"], [])


class TokenCapTests(unittest.TestCase):
    def test_a_token_cap_is_passed_to_the_client(self):
        client = FakeLLMClient(replies=["TEXT"])
        slide_vision.extract_from_images_with_report(
            ["data:image/png;base64,AAA"], client, max_tokens=1234)
        self.assertEqual(client.max_tokens_used, [1234])

    def test_the_default_cap_comes_from_configuration(self):
        import config
        client = FakeLLMClient(replies=["TEXT"])
        slide_vision.extract_from_images_with_report(
            ["data:image/png;base64,AAA"], client)
        self.assertEqual(client.max_tokens_used,
                         [config.VISION_NUM_PREDICT or None])


class BackwardCompatibilityTests(unittest.TestCase):
    def test_extract_from_images_still_returns_a_plain_string(self):
        # Existing callers and the saved output format must be unaffected,
        # or the before-and-after comparison this change enables would not
        # be measuring one thing.
        client = FakeLLMClient(replies=["PAGE ONE", "PAGE TWO"])
        text = slide_vision.extract_from_images(IMAGES, client)
        self.assertIsInstance(text, str)
        self.assertEqual(text.count(slide_vision.PAGE_MARKER), 2)

    def test_a_client_without_the_new_arguments_still_works(self):
        """A double implementing only the original signature must not break.

        Written because the real risk of adding parameters to a shared client
        is that some other caller passes an older object.
        """
        class OldStyleClient:
            def __init__(self):
                self.calls = 0

            def chat_with_images(self, prompt, images):
                self.calls += 1
                return "OLD STYLE REPLY"

        client = OldStyleClient()
        text, report = slide_vision.extract_from_images_with_report(
            ["data:image/png;base64,AAA"], client)
        self.assertEqual(client.calls, 1)
        self.assertIn("OLD STYLE REPLY", text)
        self.assertEqual(report["unreadable_pages"], [])


class DescribeReportTests(unittest.TestCase):
    def test_a_clean_report_describes_itself_as_nothing(self):
        report = {"pages": 3, "unreadable_pages": [], "truncated_pages": [],
                  "retries": 0}
        self.assertEqual(slide_vision.describe_report(report), "")

    def test_failures_are_described_in_plain_words(self):
        report = {"pages": 10, "unreadable_pages": [2, 7],
                  "truncated_pages": [4], "retries": 3}
        summary = slide_vision.describe_report(report)
        self.assertIn("2 of 10", summary)
        self.assertIn("2, 7", summary)
        self.assertIn("cut short", summary)


class InstructionTests(unittest.TestCase):
    def test_the_instruction_requires_both_sections(self):
        instruction = slide_vision.VISION_INSTRUCTION
        self.assertIn("TEXT:", instruction)
        self.assertIn("VISUALS:", instruction)

    def test_the_instruction_no_longer_offers_an_escape(self):
        """The old wording let the model skip the description entirely.

        It ended "If the slide has no diagram, transcribe the text only",
        and on a slide carrying three labelled figures the model took that
        option and returned no description at all. The replacement requires
        the section to exist and to say "none" when it is genuinely empty.
        """
        instruction = slide_vision.VISION_INSTRUCTION.lower()
        self.assertNotIn("transcribe the text only", instruction)
        self.assertIn("none", instruction)


if __name__ == "__main__":
    unittest.main()
