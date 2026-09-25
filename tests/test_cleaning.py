"""Tests for the text cleaning helpers.

Part of the context, prompts and provenance group.
"""

import unittest

from core.cleaning import clean_text, truncate_words


class CleanTextTests(unittest.TestCase):
    """Line endings are normalised, lines are stripped and blank lines go."""

    def test_strips_lines_and_drops_empty_lines(self):
        raw = "  first line  \n\n   \n  second line\n"
        self.assertEqual(clean_text(raw), "first line\nsecond line")

    def test_normalises_windows_line_endings(self):
        self.assertEqual(clean_text("a\r\nb\rc"), "a\nb\nc")

    def test_empty_and_none_return_empty_string(self):
        self.assertEqual(clean_text(""), "")
        self.assertEqual(clean_text(None), "")


class TruncateWordsTests(unittest.TestCase):
    """Text is cut to the word cap only when it is over it."""

    def test_short_text_unchanged(self):
        text, truncated = truncate_words("one two three", 5)
        self.assertEqual(text, "one two three")
        self.assertFalse(truncated)

    def test_long_text_truncated(self):
        text, truncated = truncate_words("one two three four", 2)
        self.assertEqual(text, "one two")
        self.assertTrue(truncated)


if __name__ == "__main__":
    unittest.main()
