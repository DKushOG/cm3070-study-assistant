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
