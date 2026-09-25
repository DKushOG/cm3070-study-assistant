"""Build the labelled prompt context from the sources that are available.

Each source is cleaned, given a label and joined into one block. The builder
is driven by the sources rather than by a mode name, so the caller passes
whatever it has and the mode name is worked out from that. The labels and the
joining format are kept as they were in the earlier prototype, so generation
behaves the same way across project stages.

An optional per-source word cap is applied through max_words. With max_words
set to None no cap is applied. When a cap is set, each source is shortened on
its own and every source that lost content is reported, so truncation shows up
in the interface and in the saved output instead of happening quietly.
"""
from dataclasses import dataclass

from core.cleaning import clean_text, truncate_words

TRANSCRIPT_LABEL = "[Lecture transcript]"
SLIDES_LABEL = "[Slide text]"
NOTES_LABEL = "[Student notes]"


@dataclass
class TruncationEvent:
    """One source that was shortened to fit the word cap.

    Recorded per source so the interface and the saved output can say which
    source lost content and how much.
    """
    source: str
    original_words: int
    kept_words: int

    @property
    def dropped_words(self):
        return self.original_words - self.kept_words

    def describe(self):
        return (f"{self.source} truncated: kept {self.kept_words} of "
                f"{self.original_words} words ({self.dropped_words} dropped)")


def build_context_with_report(transcript=None, slide_text=None, notes=None,
                              max_words=None):
    """Assemble the labelled context and report any truncation.

    With max_words as None no cap is applied. When it is set, each source is
    capped to at most max_words words using truncate_words, and every source
    that lost content is returned as a TruncationEvent.

    Returns (context, events). Raises ValueError when no source has any
    content, because the pipeline has nothing to generate from.
    """
    labelled_sources = [
        (TRANSCRIPT_LABEL, transcript),
        (SLIDES_LABEL, slide_text),
        (NOTES_LABEL, notes),
    ]
    sections = []
    events = []
    for label, raw in labelled_sources:
        if raw and raw.strip():
            cleaned = clean_text(raw)
            if max_words is not None:
                original_words = len(cleaned.split())
                cleaned, was_truncated = truncate_words(cleaned, max_words)
                if was_truncated:
                    events.append(TruncationEvent(
                        source=label,
                        original_words=original_words,
                        kept_words=len(cleaned.split()),
                    ))
            sections.append(label + "\n" + cleaned)
    if not sections:
        raise ValueError("At least one source must be provided.")
    return "\n\n".join(sections), events


def build_context(transcript=None, slide_text=None, notes=None):
    """Assemble the labelled context with no word cap.

    Kept with its original signature so existing callers still work. Use
    build_context_with_report to apply and report a cap. Raises ValueError
    when no source has any content.
    """
    context, _events = build_context_with_report(
        transcript=transcript, slide_text=slide_text, notes=notes)
    return context


def mode_name(transcript=None, slide_text=None, notes=None):
    """Return the input mode name for a combination of sources.

    The names used in the earlier evaluation are kept exactly, so runs from
    different project stages can be compared.
    """
    t = bool(transcript and transcript.strip())
    s = bool(slide_text and slide_text.strip())
    n = bool(notes and notes.strip())
    if t and s and n:
        return "Transcript, slide text and notes"
    if t and s:
        return "Combined transcript and slide text"
    if t and n:
        return "Transcript and notes"
    if s and n:
        return "Slide text and notes"
    if t:
        return "Transcript only"
    if s:
        return "Slide text only"
    if n:
        return "Notes only"
    return "No sources"
