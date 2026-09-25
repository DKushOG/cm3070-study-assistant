"""Tests for reading PowerPoint files.

Part of the input modules and extractor comparison group.
"""
import os
import tempfile
import unittest
from unittest import mock

from modules import slide_vision, slides_pptx


def build_deck(path, with_notes=False, with_table=False):
    """Create a small two-slide deck for testing."""
    from pptx import Presentation
    from pptx.util import Inches
    presentation = Presentation()
    layout = presentation.slide_layouts[1]  # title and content
    for index, (title, body) in enumerate(
            [("First slide title", "First slide body text"),
             ("Second slide title", "Second slide body text")]):
        slide = presentation.slides.add_slide(layout)
        slide.shapes.title.text = title
        slide.placeholders[1].text = body
        if with_notes:
            slide.notes_slide.notes_text_frame.text = f"Secret notes {index}"
    if with_table:
        slide = presentation.slides.add_slide(presentation.slide_layouts[5])
        slide.shapes.title.text = "Table slide"
        table = slide.shapes.add_table(
            2, 2, Inches(1), Inches(2), Inches(4), Inches(1)).table
        table.cell(0, 0).text = "Header A"
        table.cell(0, 1).text = "Header B"
        table.cell(1, 0).text = "Value A"
        table.cell(1, 1).text = "Value B"
    presentation.save(path)
    return path


requires_pptx = unittest.skipUnless(slides_pptx.is_available(),
                                    "python-pptx not installed")


class AvailabilityTests(unittest.TestCase):
    """The two availability checks, including the Windows install location."""

    def test_availability_flags_are_boolean(self):
        self.assertIsInstance(slides_pptx.is_available(), bool)
        self.assertIsInstance(slides_pptx.libreoffice_available(), bool)

    def test_find_soffice_returns_none_when_absent(self):
        with mock.patch.object(slides_pptx.shutil, "which",
                               return_value=None), \
                mock.patch.object(slides_pptx.os.path, "isfile",
                                  return_value=False):
            self.assertIsNone(slides_pptx.find_soffice())
            self.assertFalse(slides_pptx.libreoffice_available())

    def test_windows_install_location_is_checked_when_not_on_path(self):
        with mock.patch.object(slides_pptx.shutil, "which",
                               return_value=None), \
                mock.patch.object(slides_pptx.os.path, "isfile",
                                  return_value=True):
            self.assertEqual(slides_pptx.find_soffice(),
                             slides_pptx.WINDOWS_SOFFICE)


class LibreOfficeAbsentTests(unittest.TestCase):
    """Without LibreOffice, conversion explains what is missing."""

    def test_conversion_without_libreoffice_explains_the_fallback(self):
        """The message must name both ways forward, because this is the path a
        user without LibreOffice actually hits."""
        with mock.patch.object(slides_pptx, "find_soffice",
                               return_value=None):
            with self.assertRaises(RuntimeError) as caught:
                slides_pptx.convert_to_pdf("deck.pptx", "outdir")
        message = str(caught.exception)
        self.assertIn("LibreOffice", message)
        self.assertIn("export", message.lower())
        self.assertIn("text extraction still works", message)


@requires_pptx
class ExtractPptxTextTests(unittest.TestCase):
    """Slide text keeps its structure.

        Page markers, titles and table rows survive, and speaker notes are
        left out unless they are asked for.

    """

    def setUp(self):
        handle, self.path = tempfile.mkstemp(suffix=".pptx")
        os.close(handle)
        self.addCleanup(os.unlink, self.path)

    def test_slide_boundaries_use_the_shared_page_marker(self):
        """The same marker as the vision path, so the outputs of different
        extraction methods line up when they are compared."""
        build_deck(self.path)
        text = slides_pptx.extract_pptx_text(self.path)
        self.assertEqual(text.count(slide_vision.PAGE_MARKER), 2)
        self.assertIn(f"{slide_vision.PAGE_MARKER} 1 ---", text)
        self.assertIn(f"{slide_vision.PAGE_MARKER} 2 ---", text)

    def test_titles_and_body_text_are_extracted(self):
        build_deck(self.path)
        text = slides_pptx.extract_pptx_text(self.path)
        self.assertIn("First slide title", text)
        self.assertIn("First slide body text", text)
        self.assertIn("Second slide title", text)

    def test_title_is_not_duplicated(self):
        build_deck(self.path)
        text = slides_pptx.extract_pptx_text(self.path)
        self.assertEqual(text.count("First slide title"), 1)

    def test_table_cells_are_extracted_with_row_structure(self):
        build_deck(self.path, with_table=True)
        text = slides_pptx.extract_pptx_text(self.path)
        self.assertIn("Header A\tHeader B", text)
        self.assertIn("Value A\tValue B", text)

    def test_speaker_notes_are_excluded_by_default(self):
        """Off by default is a fairness requirement: no other extraction path
        can see speaker notes, so including them would confound the
        comparison."""
        build_deck(self.path, with_notes=True)
        text = slides_pptx.extract_pptx_text(self.path)
        self.assertNotIn("Secret notes", text)

    def test_speaker_notes_included_only_when_requested(self):
        build_deck(self.path, with_notes=True)
        text = slides_pptx.extract_pptx_text(self.path, include_notes=True)
        self.assertIn("Secret notes 0", text)
        self.assertIn(slides_pptx.NOTES_MARKER, text)


if __name__ == "__main__":
    unittest.main()
