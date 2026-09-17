"""Builds the labelled prompt context from the available sources.

The labels and joining format are identical to the evaluated prototype so
that generation behaviour carries over unchanged. The builder is source
driven rather than mode driven: pass whichever sources exist and the mode
name is derived from them.

build_context_with_report adds an optional per-source word cap for long
inputs. It is opt-in through max_words: when max_words is None the output is
byte-identical to the original builder, and build_context is a thin wrapper
over it for that no-limit case. When a cap is set, each source is truncated
independently and every source that lost content is reported, so truncation
is visible in the interface and in the saved output rather than silent.
"""
from dataclasses import dataclass

from core.cleaning import clean_text, truncate_words

TRANSCRIPT_LABEL = "[Lecture transcript]"
SLIDES_LABEL = "[Slide text]"
NOTES_LABEL = "[Student notes]"


@dataclass
class TruncationEvent:
    """One source that was shortened to fit the word cap.

    Recorded per source so the interface and the saved output can state
    exactly which source lost content and how much, which keeps truncation
    visible rather than silent (the transparency requirement in the Design
    chapter).
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

    When max_words is None there is no limit and the assembled context is
    byte-identical to the original builder. When max_words is set, each
    source is independently capped to at most max_words words using
    truncate_words, and every source that actually lost content is returned
    as a TruncationEvent.

    Returns (context, events) where events is a list of TruncationEvent.
    Raises ValueError if no source has any content, because the pipeline has
    nothing to generate from.
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
    """Assemble the labelled context from the provided sources.

    This is the unlimited path, preserved with its original signature and
    behaviour; use build_context_with_report to apply and report a word cap.
    Raises ValueError if no source has any content.
    """
    context, _events = build_context_with_report(
        transcript=transcript, slide_text=slide_text, notes=notes)
    return context


def mode_name(transcript=None, slide_text=None, notes=None):
    """Return the canonical input mode name for a source combination.

    The four names used in the Preliminary Project Report evaluation are
    preserved exactly so results can be compared across project stages.
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
