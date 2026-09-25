"""Tests for the schema-constrained quiz path.

Part of the structured quiz schema group.
"""
import json
import unittest

from core.context_builder import build_context
from core.generation import (MIN_ANSWER_CHARS, MIN_QUESTION_CHARS,
                             QUESTION_TYPES, available_basis_choices,
                             build_quiz_schema, generate_quiz_structured,
                             render_quiz_text)
from core.quiz_validation import validate_quiz
from tests.fakes import FakeLLMClient


def quiz_payload(basis="transcript", count=5, question_type="definition"):
    return {"questions": [
        {
            "question": f"What is concept {index} and why does it matter?",
            "suggested_answer": f"Concept {index} is defined in the source.",
            "question_type": question_type,
            "source_basis": basis,
        }
        for index in range(1, count + 1)
    ]}


class BasisChoicesTest(unittest.TestCase):
    """Only the sources this run supplied are offered as a basis.

        The enum, the length floors and the question count all come from the
        schema, so a reply that breaks them is not one the model can produce.

    """

    def test_only_supplied_sources_are_offered(self):
        context = build_context(transcript="spoken words",
                                slide_text="slide words")
        self.assertEqual(available_basis_choices(context),
                         ["transcript", "slide text"])

    def test_student_notes_offered_only_when_notes_supplied(self):
        without = build_context(transcript="spoken words")
        self.assertNotIn("student notes", available_basis_choices(without))
        with_notes = build_context(transcript="spoken words",
                                   notes="my own notes")
        self.assertIn("student notes", available_basis_choices(with_notes))

    def test_schema_enum_matches_the_available_sources(self):
        context = build_context(slide_text="slide words")
        schema = build_quiz_schema(available_basis_choices(context))
        item = schema["properties"]["questions"]["items"]
        self.assertEqual(item["properties"]["source_basis"]["enum"],
                         ["slide text"])
        self.assertEqual(item["properties"]["question_type"]["enum"],
                         list(QUESTION_TYPES))
        self.assertEqual(sorted(item["required"]),
                         ["question", "question_type", "source_basis",
                          "suggested_answer"])

    def test_free_text_fields_carry_a_length_floor(self):
        """A required string property is satisfied by an empty string.

        Two runs in the first structured campaign returned five well-formed
        questions whose suggested answers were all empty, so the floor is a
        measured requirement rather than a precaution.
        """
        item = build_quiz_schema(["transcript"])["properties"]["questions"]["items"]
        self.assertEqual(item["properties"]["question"]["minLength"],
                         MIN_QUESTION_CHARS)
        self.assertEqual(item["properties"]["suggested_answer"]["minLength"],
                         MIN_ANSWER_CHARS)

    def test_schema_fixes_the_question_count(self):
        schema = build_quiz_schema(["transcript"], count=5)
        questions = schema["properties"]["questions"]
        self.assertEqual(questions["minItems"], 5)
        self.assertEqual(questions["maxItems"], 5)


class RenderedOutputTest(unittest.TestCase):
    """A structured reply renders into the text form the validator parses."""

    def test_rendered_quiz_passes_the_existing_validator(self):
        text = render_quiz_text(quiz_payload())
        result = validate_quiz(text)
        self.assertTrue(result.passed, result.failures)

    def test_rendered_quiz_carries_all_four_labels_per_question(self):
        text = render_quiz_text(quiz_payload(count=2))
        for label in ("Question:", "Suggested answer:", "Question type:",
                      "Source basis:"):
            self.assertEqual(text.count(label), 2, label)

    def test_validator_still_catches_a_generated_notes_attribution(self):
        """The validator must not have been weakened by any of this.

        Rendering a payload that credits the generated revision notes has to
        keep failing, otherwise the drop in that fault count would be an
        artefact of a softened check rather than a real improvement.
        """
        text = render_quiz_text(quiz_payload(basis="revision notes"))
        result = validate_quiz(text)
        self.assertFalse(result.passed)

    def test_empty_payload_renders_empty_rather_than_raising(self):
        self.assertEqual(render_quiz_text({}), "")


class GenerationPathTest(unittest.TestCase):
    """The schema reaches the model and a bad reply is handled.

        Malformed JSON returns an empty string rather than raising, so one bad
        reply is scored as a failed run instead of stopping a campaign.

    """

    def test_structured_call_attaches_a_json_schema(self):
        client = FakeLLMClient(replies=[json.dumps(quiz_payload())])
        context = build_context(transcript="spoken words")
        generate_quiz_structured(client, context, "some notes")
        sent = client.response_formats[-1]
        self.assertEqual(sent["type"], "json_schema")
        self.assertEqual(sent["json_schema"]["name"], "quiz")
        self.assertTrue(sent["json_schema"]["strict"])

    def test_enum_sent_to_the_model_excludes_the_generated_notes(self):
        client = FakeLLMClient(replies=[json.dumps(quiz_payload())])
        context = build_context(transcript="spoken words",
                                slide_text="slide words")
        generate_quiz_structured(client, context, "some notes")
        enum = (client.response_formats[-1]["json_schema"]["schema"]
                ["properties"]["questions"]["items"]["properties"]
                ["source_basis"]["enum"])
        self.assertNotIn("revision notes", enum)
        self.assertNotIn("student notes", enum)
        self.assertEqual(enum, ["transcript", "slide text"])

    def test_structured_output_validates_end_to_end(self):
        client = FakeLLMClient(replies=[json.dumps(quiz_payload())])
        context = build_context(transcript="spoken words")
        text = generate_quiz_structured(client, context, "some notes")
        self.assertTrue(validate_quiz(text).passed)

    def test_malformed_json_returns_empty_rather_than_raising(self):
        client = FakeLLMClient(replies=["not json at all {"])
        context = build_context(transcript="spoken words")
        self.assertEqual(
            generate_quiz_structured(client, context, "some notes"), "")

    def test_prompt_names_only_the_supplied_sources(self):
        client = FakeLLMClient(replies=[json.dumps(quiz_payload())])
        context = build_context(slide_text="slide words", notes="my notes")
        generate_quiz_structured(client, context, "some notes")
        prompt = client.prompts[-1]
        self.assertIn("slide text, student notes", prompt)
        self.assertIn("must never be given as the source basis", prompt)


if __name__ == "__main__":
    unittest.main()
