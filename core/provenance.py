"""Track the source of extracted text.

Showing which model produced which text is one of the project's aims, so a
caption that credits the wrong thing is a real fault. The interface records
the exact text an extraction produced. These functions compare that against
what is in the editor now and return the label to show.

The rule is credit, then qualify. An edit after extraction is usually a small
correction, so the extractor is still the origin of most of the text. Keeping
the name and adding "(edited after extraction)" tells the reader both facts,
where dropping the name would hide which model did the work and would make a
corrected extraction look like hand-typed text in the evaluation tables.

The same check serves slide text, where the source is the extraction path, and
transcripts, where it is the Whisper size. Two named wrappers cover the two
cases rather than repeating the logic.

Kept in core as pure functions so it can be tested without importing the
Streamlit app, which would run the whole interface script.
"""

EDITED_SUFFIX = " (edited after extraction)"


def describe_provenance(current_text, source, produced_text):
    """Return the label to display and record, or None when there is none.

    None is returned when the text box is empty or cleared, and when nothing
    produced the text, since pasted or typed text has no extractor to credit.
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
    """Label for the transcript: which Whisper size produced it.

    The same audio is run through several Whisper sizes during evaluation, so
    a label left behind from an earlier size would credit the transcript to the
    wrong model.
    """
    source = f"Whisper {model_size}" if model_size else None
    return describe_provenance(current_text, source, transcribed_text)
