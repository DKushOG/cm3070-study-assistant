"""Text provenance.

Which model produced which text is a core transparency principle of this
project, so a provenance caption that can lie is a real defect. The interface
records the exact text an extraction produced; this pure function compares
that against what is in the editor now and decides what may honestly be
claimed.

The rule is source-agnostic, so the same function serves both the slide text
(which extraction path produced it) and the transcript (which Whisper model
size produced it). Two named wrappers document the two cases rather than
duplicating the logic, because the staleness problem is identical: any editable
box whose contents are credited to a model can drift from what that model
actually produced.

The rule is deliberately "credit, then qualify" rather than "drop the credit":
an edit after extraction is usually a small correction of the extractor's
output, so the extractor is still the origin of most of the text. Saying
"edited after extraction by X" tells the reader both facts, where dropping the
credit entirely would hide which model did the work and would make a corrected
vision extraction indistinguishable from hand-typed text in the evaluation
tables. It never overclaims, because the edit is always disclosed.

Kept in core as a pure function so it is unit testable without importing the
Streamlit app, which would execute the whole interface script.
"""

EDITED_SUFFIX = " (edited after extraction)"


def describe_provenance(current_text, source, produced_text):
    """Return the provenance to display and record, or None if none can be
    honestly claimed.

    None is returned when the text area is empty or cleared, or when nothing
    produced the text (manually pasted or typed text claims nothing).
    """
    if not source or not (current_text or "").strip():
        return None
    if current_text == produced_text:
        return source
    return source + EDITED_SUFFIX


def describe_slide_provenance(current_text, source, extracted_text):
    """Provenance for slide text: which extraction path produced it."""
    return describe_provenance(current_text, source, extracted_text)


def describe_transcript_provenance(current_text, model_size, transcribed_text):
    """Provenance for the transcript: which Whisper model size produced it.

    Needed because the Evaluation chapter runs the same audio through several
    Whisper sizes, so a stale size label would misattribute a transcript to the
    wrong model and corrupt the word error rate comparison.
    """
    source = f"Whisper {model_size}" if model_size else None
    return describe_provenance(current_text, source, transcribed_text)
