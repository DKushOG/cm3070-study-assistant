"""Text cleaning helpers.

Pure functions with no dependencies, so they can be tested without a model or a
network. clean_text keeps the behaviour of the prototype, so results stay
comparable between project stages.
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
