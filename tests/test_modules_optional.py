import unittest

from modules import audio_stt, slides_ocr


class AvailabilityTests(unittest.TestCase):
    def test_availability_flags_are_boolean(self):
        self.assertIsInstance(audio_stt.is_available(), bool)
        self.assertIsInstance(slides_ocr.pdf_available(), bool)
        self.assertIsInstance(slides_ocr.ocr_available(), bool)


class AcceptedAudioFormatTests(unittest.TestCase):
    """The uploader's accepted list is the thing that rejected a real lecture
    recording, so it is pinned here against the requirement in Table 3.1."""

    def test_common_lecture_containers_are_accepted(self):
        for fmt in ("mp3", "wav", "m4a", "mp4", "webm", "flac", "ogg"):
            self.assertIn(fmt, audio_stt.ACCEPTED_AUDIO_FORMATS)

    def test_mp4_is_accepted_because_lectures_arrive_as_video(self):
        self.assertIn("mp4", audio_stt.ACCEPTED_AUDIO_FORMATS)

    def test_formats_are_bare_extensions_without_dots(self):
        """Streamlit's file_uploader expects extensions without a leading
        dot, so a stray dot would silently reject every file."""
        for fmt in audio_stt.ACCEPTED_AUDIO_FORMATS:
            self.assertFalse(fmt.startswith("."))
            self.assertEqual(fmt, fmt.lower())

    def test_whisper_sizes_cover_the_word_error_rate_comparison(self):
        for size in ("tiny", "base", "small"):
            self.assertIn(size, audio_stt.WHISPER_MODEL_SIZES)


class ExtractSlidesDispatchTests(unittest.TestCase):
    def test_unsupported_extension_rejected(self):
        with self.assertRaises(ValueError):
            slides_ocr.extract_slides("lecture.txt")


@unittest.skipUnless(audio_stt.is_available(),
                     "openai-whisper not installed")
class WhisperSmokeTests(unittest.TestCase):
    def test_whisper_module_loads(self):
        import whisper
        self.assertTrue(hasattr(whisper, "load_model"))


if __name__ == "__main__":
    unittest.main()
