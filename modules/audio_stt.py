"""Speech to text with Whisper.

Wrapped so the app still runs when Whisper or ffmpeg is missing, in which case
the user pastes a transcript instead. Video containers are accepted because a
lecture recording is usually a video file, and ffmpeg reads the audio from it.
"""

# Containers ffmpeg can decode. Video files are included on purpose.
ACCEPTED_AUDIO_FORMATS = ["mp3", "wav", "m4a", "mp4", "webm", "flac", "ogg"]

# Whisper sizes available in the app, ordered from smallest to largest. The
# evaluation compares tiny, base and small.
WHISPER_MODEL_SIZES = ["tiny", "base", "small", "medium"]


def is_available():
    try:
        import whisper  # noqa: F401
        return True
    except ImportError:
        return False


def transcribe(audio_path, model_size=None):
    """Transcribe an audio file to text using Whisper.

    Raises RuntimeError with an installation hint if Whisper is missing.
    """
    try:
        import whisper
    except ImportError as error:
        raise RuntimeError(
            "openai-whisper is not installed. Install it with "
            "'pip install openai-whisper' and make sure ffmpeg is on the "
            "system path."
        ) from error
    import config
    model = whisper.load_model(model_size or config.WHISPER_MODEL)
    result = model.transcribe(str(audio_path))
    return (result.get("text") or "").strip()
