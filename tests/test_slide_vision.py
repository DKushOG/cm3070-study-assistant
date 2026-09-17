"""Tests for the vision-based slide extraction module.

These use the injected FakeLLMClient and, where a file is unavoidable, a
throwaway temp file whose bytes need not be a real image (encoding only
base64s the bytes). No test makes a real model call, touches the network, or
requires PyMuPDF: the one rendering test runs only when PyMuPDF is absent, to
prove the graceful-degradation path.
"""
import base64
import os
import tempfile
import unittest

from modules import slide_vision
from tests.fakes import FakeLLMClient


class InstructionTests(unittest.TestCase):
    def test_instruction_asks_for_verbatim_text_and_diagrams(self):
        text = slide_vision.VISION_INSTRUCTION.lower()
        self.assertIn("verbatim", text)
        self.assertIn("diagram", text)


class AvailabilityTests(unittest.TestCase):
    def test_is_available_is_boolean(self):
        self.assertIsInstance(slide_vision.is_available(), bool)


class ExtractDispatchTests(unittest.TestCase):
    def test_unsupported_extension_rejected(self):
        with self.assertRaises(ValueError):
            slide_vision.extract_slides("lecture.txt", FakeLLMClient())


class ExtractFromImagesTests(unittest.TestCase):
    def test_one_call_per_image_with_page_markers(self):
        client = FakeLLMClient(replies=["PAGE ONE TEXT", "PAGE TWO TEXT"])
        images = ["data:image/png;base64,AAA", "data:image/png;base64,BBB"]
        text = slide_vision.extract_from_images(images, client)
        self.assertEqual(len(client.image_calls), 2)
        # The quotable instruction and the exact image URL are passed through.
        self.assertEqual(client.image_calls[0][0],
                         slide_vision.VISION_INSTRUCTION)
        self.assertEqual(client.image_calls[0][1],
                         ["data:image/png;base64,AAA"])
        self.assertIn("PAGE ONE TEXT", text)
        self.assertIn("PAGE TWO TEXT", text)
        self.assertEqual(text.count(slide_vision.PAGE_MARKER), 2)


class ExtractImageFileTests(unittest.TestCase):
    def test_image_file_is_encoded_and_sent(self):
        client = FakeLLMClient(replies=["SLIDE TEXT"])
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(b"\x89PNG-fake-bytes")
            path = tmp.name
        try:
            text = slide_vision.extract_slides(path, client)
        finally:
            os.unlink(path)
        self.assertEqual(len(client.image_calls), 1)
        url = client.image_calls[0][1][0]
        self.assertTrue(url.startswith("data:image/png;base64,"))
        payload = url.split(",", 1)[1]
        self.assertEqual(base64.b64decode(payload), b"\x89PNG-fake-bytes")
        self.assertIn("SLIDE TEXT", text)

    def test_jpg_uses_jpeg_mime(self):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(b"jpeg-fake-bytes")
            path = tmp.name
        try:
            url = slide_vision.encode_image_file(path)
        finally:
            os.unlink(path)
        self.assertTrue(url.startswith("data:image/jpeg;base64,"))


class RenderDegradationTests(unittest.TestCase):
    @unittest.skipIf(slide_vision.is_available(),
                     "PyMuPDF is installed, so the missing-dependency path "
                     "cannot be exercised")
    def test_pdf_render_without_pymupdf_raises_runtimeerror(self):
        with self.assertRaises(RuntimeError):
            slide_vision.render_pdf_to_images("deck.pdf")


if __name__ == "__main__":
    unittest.main()
