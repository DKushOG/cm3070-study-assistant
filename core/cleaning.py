"""Text cleaning utilities.

Pure functions with no external dependencies so they can be unit tested
without any model or network access. clean_text preserves the behaviour of
the evaluated Preliminary Project Report prototype so results remain
comparable across project stages.
"""


def clean_text(text):
    """Basic text cleaning: normalise line endings, strip each line and
    drop empty lines."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def truncate_words(text, max_words):
    """Limit text to max_words. Returns (text, was_truncated)."""
    words = text.split()
    if len(words) <= max_words:
        return text, False
    return " ".join(words[:max_words]), True
