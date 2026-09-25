"""Tests for the source label shown with extracted text.

Part of the context, prompts and provenance group.
"""
import unittest

from core.provenance import (EDITED_SUFFIX, describe_slide_provenance,
                             describe_transcript_provenance)

SOURCE = "Vision model (qwen3-vl:4b)"


class DescribeSlideProvenanceTests(unittest.TestCase):
    """The extractor is credited, and an edit after extraction is disclosed."""

    def test_unedited_extraction_is_credited_plainly(self):
        self.assertEqual(
            describe_slide_provenance("extracted text", SOURCE,
                                      "extracted text"),
            SOURCE)

    def test_edited_text_discloses_the_edit(self):
        result = describe_slide_provenance("extracted text, corrected",
                                           SOURCE, "extracted text")
        self.assertEqual(result, SOURCE + EDITED_SUFFIX)
        self.assertIn(SOURCE, result)

    def test_cleared_text_claims_nothing(self):
        self.assertIsNone(
            describe_slide_provenance("", SOURCE, "extracted text"))
        self.assertIsNone(
            describe_slide_provenance("   ", SOURCE, "extracted text"))

    def test_wholesale_replacement_is_disclosed_not_credited_silently(self):
        result = describe_slide_provenance("completely different text",
                                           SOURCE, "extracted text")
        self.assertIn(EDITED_SUFFIX.strip(), result)

    def test_manually_pasted_text_claims_nothing(self):
        self.assertIsNone(describe_slide_provenance("pasted by hand", "", ""))
        self.assertIsNone(
            describe_slide_provenance("pasted by hand", None, ""))


class TranscriptProvenanceTests(unittest.TestCase):
    """The Whisper size label follows the text it came from.

        A label left behind from an earlier size would credit a transcript to
        the wrong model in the word error rate comparison.

    """

    def test_names_the_whisper_size(self):
        self.assertEqual(
            describe_transcript_provenance("spoken words", "small",
                                           "spoken words"),
            "Whisper small")

    def test_edited_transcript_discloses_the_edit(self):
        result = describe_transcript_provenance("spoken words, fixed", "tiny",
                                                "spoken words")
        self.assertEqual(result, "Whisper tiny" + EDITED_SUFFIX)

    def test_pasted_transcript_claims_no_model(self):
        self.assertIsNone(
            describe_transcript_provenance("typed by hand", "", ""))

    def test_cleared_transcript_claims_nothing(self):
        self.assertIsNone(
            describe_transcript_provenance("", "base", "spoken words"))


if __name__ == "__main__":
    unittest.main()
