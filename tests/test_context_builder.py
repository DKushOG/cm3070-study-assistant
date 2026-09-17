import unittest

from core.context_builder import (NOTES_LABEL, SLIDES_LABEL,
                                  TRANSCRIPT_LABEL, build_context,
                                  build_context_with_report, mode_name)


class BuildContextTests(unittest.TestCase):
    def test_all_sources_labelled(self):
        context = build_context(transcript="t", slide_text="s", notes="n")
        for label in (TRANSCRIPT_LABEL, SLIDES_LABEL, NOTES_LABEL):
            self.assertIn(label, context)

    def test_only_provided_sources_included(self):
        context = build_context(slide_text="slides here")
        self.assertIn(SLIDES_LABEL, context)
        self.assertNotIn(TRANSCRIPT_LABEL, context)
        self.assertNotIn(NOTES_LABEL, context)

    def test_sources_appear_in_fixed_order(self):
        context = build_context(transcript="t", slide_text="s", notes="n")
        self.assertLess(context.index(TRANSCRIPT_LABEL),
                        context.index(SLIDES_LABEL))
        self.assertLess(context.index(SLIDES_LABEL),
                        context.index(NOTES_LABEL))

    def test_blank_sources_raise_value_error(self):
        with self.assertRaises(ValueError):
            build_context(transcript="   ", slide_text="", notes=None)

    def test_content_is_cleaned(self):
        context = build_context(transcript="  line one  \r\n\r\n line two ")
        self.assertIn("line one\nline two", context)


class BuildContextWithReportTests(unittest.TestCase):
    def test_no_limit_is_byte_identical_and_reports_nothing(self):
        kwargs = dict(transcript="a b c", slide_text="d e", notes="f")
        context, events = build_context_with_report(**kwargs)
        self.assertEqual(context, build_context(**kwargs))
        self.assertEqual(events, [])

    def test_none_max_words_never_truncates(self):
        context, events = build_context_with_report(
            transcript="one two three four five", max_words=None)
        self.assertIn("five", context)
        self.assertEqual(events, [])

    def test_truncates_only_the_oversized_source(self):
        context, events = build_context_with_report(
            transcript="one two three four five",
            notes="short notes",
            max_words=2)
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event.source, TRANSCRIPT_LABEL)
        self.assertEqual(event.original_words, 5)
        self.assertEqual(event.kept_words, 2)
        self.assertEqual(event.dropped_words, 3)
        self.assertIn("one two", context)
        self.assertNotIn("three", context)
        # A source within the limit is untouched and not reported.
        self.assertIn("short notes", context)

    def test_each_truncated_source_is_reported(self):
        _context, events = build_context_with_report(
            transcript="a a a a", slide_text="b b b b", max_words=1)
        self.assertEqual({event.source for event in events},
                         {TRANSCRIPT_LABEL, SLIDES_LABEL})

    def test_describe_names_source_and_counts(self):
        _context, events = build_context_with_report(
            transcript="one two three", max_words=1)
        text = events[0].describe()
        self.assertIn(TRANSCRIPT_LABEL, text)
        self.assertIn("2 dropped", text)

    def test_blank_sources_raise_value_error(self):
        with self.assertRaises(ValueError):
            build_context_with_report(transcript="  ", max_words=10)


class ModeNameTests(unittest.TestCase):
    def test_four_canonical_modes(self):
        self.assertEqual(mode_name(transcript="t"), "Transcript only")
        self.assertEqual(mode_name(slide_text="s"), "Slide text only")
        self.assertEqual(mode_name(transcript="t", slide_text="s"),
                         "Combined transcript and slide text")
        self.assertEqual(
            mode_name(transcript="t", slide_text="s", notes="n"),
            "Transcript, slide text and notes")

    def test_no_sources(self):
        self.assertEqual(mode_name(), "No sources")


if __name__ == "__main__":
    unittest.main()
