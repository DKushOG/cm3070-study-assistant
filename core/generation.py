"""Prompt text and the two generation calls.

The prompts are copied from the earlier prototype, so generation behaves the
same way as the version that was evaluated. The second call depends on the
first: the quiz is generated from the source context and the revision notes
together.
"""

REVISION_NOTES_PROMPT = """
You are helping create revision material for a university student.

Use only the information explicitly stated in the source material below.
Do not add outside facts, definitions or examples that are not present in the source.
If something is only named in the source but not explained, do not invent an explanation for it.

Output only the revision notes. Do not repeat the source material back to me, do not write a preamble and do not add any commentary about the notes after them.

Structure the revision notes with these sections:
1. Topic overview
2. Key concepts
3. Important definitions
4. Step-by-step explanation
5. Common confusion points
6. Short summary

Source material:
{context}
"""

QUIZ_PROMPT = """
You are helping create quiz questions for student revision.

Use only the source material and revision notes provided below.
Do not introduce facts that are not present in either of them.

Create exactly five open short-answer questions. Do not write multiple choice questions and do not provide answer options.

For each question, output these four labelled lines and nothing else:
Question: the question text
Suggested answer: a concise answer that is supported by the source
Question type: choose exactly one of definition, explanation, comparison or application
Source basis: state whether the answer came mainly from the transcript, the slide text or the notes

Source material:
{context}

Revision notes:
{revision_notes}
"""


def generate_revision_notes(client, context):
    return client.chat(REVISION_NOTES_PROMPT.format(context=context))


def generate_quiz_questions(client, context, revision_notes):
    return client.chat(
        QUIZ_PROMPT.format(context=context, revision_notes=revision_notes)
    )


# ---------------------------------------------------------------------------
# Structured quiz generation
# ---------------------------------------------------------------------------
# The prompt-and-check path above asks for four labelled lines and checks the
# reply afterwards. Across the recorded campaigns it passed 4 of 48 runs, none
# of them under greedy decoding, and the most common fault was one missing
# line ("Question type:") rather than a missing or malformed question. A
# prompt on its own has no way to stop the decoder skipping a line it was only
# asked for.
#
# The path below moves the requirement from the prompt into the decoder. Each
# field is a required property of a JSON schema, so a missing field is not a
# reply the model can produce. The result is rendered back into the same
# four-line text, which leaves quiz_validation, the saved output format and
# the interface unchanged and keeps both paths checked by the same code.

import json

from core.context_builder import (TRANSCRIPT_LABEL, SLIDES_LABEL, NOTES_LABEL)

# The basis value offered for each source block, paired with the label the
# context builder uses for it. The wording matters. quiz_validation checks the
# generated-notes terms before the plain "notes" term, so "student notes"
# resolves to the student notes input rather than to the system's own notes.
BASIS_CHOICES = (
    (TRANSCRIPT_LABEL, "transcript"),
    (SLIDES_LABEL, "slide text"),
    (NOTES_LABEL, "student notes"),
)

QUESTION_TYPES = ("definition", "explanation", "comparison", "application")

# Minimum lengths for the two free-text fields. A required property of type
# "string" is satisfied by an empty string, and the model does take that
# option. An early structured run returned five well-formed questions in which
# every suggested answer was empty. Requiring a field to exist is not the same
# as requiring it to say anything, and the length floor closes that gap.
MIN_QUESTION_CHARS = 10
MIN_ANSWER_CHARS = 20

# Kept separate from QUIZ_PROMPT rather than replacing it. quiz_validation
# reads QUIZ_PROMPT at import time to learn the placeholder wording it looks
# for, so editing that constant would change the validator as well, and the
# original path has to stay runnable to provide the earlier measurement.
QUIZ_SCHEMA_PROMPT = """
You are helping create quiz questions for student revision.

Use only the source material and revision notes provided below.
Do not introduce facts that are not present in either of them.

Create exactly five open short-answer questions. Do not write multiple choice questions and do not provide answer options.

Set source_basis to the supplied source block the answer came mainly from. The supplied source blocks are: {source_list}. The revision notes below were produced by this system from those blocks. They are not a source block and must never be given as the source basis.

Source material:
{context}

Revision notes:
{revision_notes}
"""


def available_basis_choices(context):
    """Return the basis values a quiz may use for this run.

    Taken from the source labels present in the assembled context, so a run
    without student notes does not offer "student notes" as a choice.
    Narrowing the enum this way keeps an attribution to the generated revision
    notes out of the reply rather than catching it afterwards.
    """
    return [value for label, value in BASIS_CHOICES if label in context]


def build_quiz_schema(basis_choices, count=5):
    """Return the JSON schema that one quiz response must satisfy."""
    return {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "minItems": count,
                "maxItems": count,
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "minLength": MIN_QUESTION_CHARS,
                        },
                        "suggested_answer": {
                            "type": "string",
                            "minLength": MIN_ANSWER_CHARS,
                        },
                        "question_type": {
                            "type": "string",
                            "enum": list(QUESTION_TYPES),
                        },
                        "source_basis": {
                            "type": "string",
                            "enum": list(basis_choices),
                        },
                    },
                    "required": ["question", "suggested_answer",
                                 "question_type", "source_basis"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["questions"],
        "additionalProperties": False,
    }


def render_quiz_text(payload):
    """Render a structured quiz into the four-labelled-line text form.

    The same shape quiz_validation already parses, so the validator is left
    alone and both paths are checked by the same code.
    """
    lines = []
    for item in payload.get("questions", []):
        lines.append("Question: " + str(item.get("question", "")).strip())
        lines.append("Suggested answer: "
                     + str(item.get("suggested_answer", "")).strip())
        lines.append("Question type: "
                     + str(item.get("question_type", "")).strip())
        lines.append("Source basis: "
                     + str(item.get("source_basis", "")).strip())
        lines.append("")
    return "\n".join(lines).strip()


def generate_quiz_structured(client, context, revision_notes, count=5):
    """Generate one quiz under a JSON schema and render it to the text form.

    A reply that is not valid JSON returns an empty string rather than raising,
    so one bad reply is scored as a failed run instead of stopping a campaign
    part way through.
    """
    choices = available_basis_choices(context)
    schema = build_quiz_schema(choices, count=count)
    prompt = QUIZ_SCHEMA_PROMPT.format(
        context=context,
        revision_notes=revision_notes,
        source_list=", ".join(choices),
    )
    raw = client.chat(prompt, response_format={
        "type": "json_schema",
        "json_schema": {"name": "quiz", "schema": schema, "strict": True},
    })
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError):
        return ""
    if not isinstance(payload, dict):
        return ""
    return render_quiz_text(payload)
