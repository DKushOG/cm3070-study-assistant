"""Prompt construction and the two sequential generation steps.

The prompt text is copied verbatim from the refined prototype evaluated in
the Preliminary Project Report, so generation behaviour is unchanged. The
second call deliberately depends on the output of the first: the quiz is
generated from the source context plus the revision notes.
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
# The prompt-and-check path above asks for four labelled lines and is checked
# afterwards. Measured over twelve runs per condition it never produced a
# fully valid quiz under greedy decoding, and the dominant fault was a single
# omitted line ("Question type:") rather than a malformed or missing question.
# That is a failure a prompt cannot reliably prevent, because nothing stops
# the decoder skipping a line it was merely asked for.
#
# The path below moves the requirement out of the prompt and into the
# decoder. Each field is a required property of a JSON schema, so an omitted
# field is not a response the model is able to emit. The result is rendered
# back into the same four-line text the validator already parses, which keeps
# quiz_validation, the saved output format and the interface unchanged, and
# therefore keeps the two paths measurable by the same instrument.

import json

from core.context_builder import (TRANSCRIPT_LABEL, SLIDES_LABEL, NOTES_LABEL)

# The basis value offered for each source block, paired with the label the
# context builder uses for it. The wording matters: quiz_validation checks
# GENERATED_NOTES_TERMS ("revision notes", "generated notes") before it checks
# the plain "notes" term, so "student notes" resolves to the student notes
# input and is not mistaken for the system's own generated notes.
BASIS_CHOICES = (
    (TRANSCRIPT_LABEL, "transcript"),
    (SLIDES_LABEL, "slide text"),
    (NOTES_LABEL, "student notes"),
)

QUESTION_TYPES = ("definition", "explanation", "comparison", "application")

# Minimum lengths for the two free-text fields. A required property of type
# "string" is satisfied by an empty string, and the model does take that
# option: under sampling, two runs in the first structured campaign returned
# five well-formed questions in which every suggested answer was "". Requiring
# the field to exist is therefore not the same as requiring it to say
# anything, and the length floor is what closes that gap.
MIN_QUESTION_CHARS = 10
MIN_ANSWER_CHARS = 20

# Kept separate from QUIZ_PROMPT rather than replacing it. quiz_validation
# parses QUIZ_PROMPT at import time to learn the placeholder wording it checks
# for, so editing that constant would silently change the validator, and the
# legacy path has to stay runnable to provide the "before" measurement.
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

    Derived from the source labels actually present in the assembled context,
    so a run without student notes cannot offer "student notes" as an answer.
    Narrowing the enum this way is what makes an attribution to the generated
    revision notes unrepresentable rather than merely detectable afterwards.
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

    Deliberately the same shape quiz_validation already parses, so the
    validator is not modified and the before-and-after comparison is measured
    by one unchanged instrument.
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

    A response that is not valid JSON returns an empty string rather than
    raising, so one bad response is scored as a failed run by the validator
    instead of ending a twelve-run campaign part way through.
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
