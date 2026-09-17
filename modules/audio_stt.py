"""Speech-to-text module (Phase A extension point).

Wraps openai-whisper behind a small interface with graceful degradation:
if the package or ffmpeg is missing, the app still runs and the user can
paste a transcript manually. That keeps the system demonstrable at every
stage of development, which matches the staged plan in the Design chapter.

Accepted formats are listed here rather than inline in the interface, so the
supported set is stated in one place and is testable. The list is wide because
Whisper hands decoding to ffmpeg, which reads every one of these containers:
the earlier mp3/wav/m4a-only restriction was an interface limitation, not a
model limitation, and it rejected mp4 — the dominant lecture recording
container. Table 3.1 of the Design chapter promises to accept mixed lecture
materials and Chapter 1.3 promises common learning material formats, so
requiring a student to convert a recording before use was a defect against a
stated requirement rather than an acceptable limitation. Video containers are
included deliberately: ffmpeg simply ignores the video stream and decodes the
audio, so no conversion step is needed.
"""

# Containers ffmpeg can decode for Whisper. Video containers (mp4, webm) are
# accepted because a lecture recording usually arrives as video.
ACCEPTED_AUDIO_FORMATS = ["mp3", "wav", "m4a", "mp4", "webm", "flac", "ogg"]

# The Whisper sizes compared for the word error rate evidence in the
# Evaluation chapter. Ordered smallest to largest, which is also increasing
# accuracy and increasing runtime.
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
