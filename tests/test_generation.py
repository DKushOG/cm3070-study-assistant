import unittest

from core import generation
from tests.fakes import FakeLLMClient


class NotesPromptTests(unittest.TestCase):
    def test_prompt_contains_context_and_constraints(self):
        client = FakeLLMClient()
        generation.generate_revision_notes(client, "THE CONTEXT")
        prompt = client.prompts[0]
        self.assertIn("THE CONTEXT", prompt)
        self.assertIn("Do not add outside facts", prompt)
        self.assertIn("Output only the revision notes", prompt)


class QuizPromptTests(unittest.TestCase):
    def test_prompt_contains_context_and_notes(self):
        client = FakeLLMClient()
        generation.generate_quiz_questions(client, "THE CONTEXT",
                                           "THE NOTES")
        prompt = client.prompts[0]
        self.assertIn("THE CONTEXT", prompt)
        self.assertIn("THE NOTES", prompt)

    def test_prompt_enforces_format_rules(self):
        self.assertIn("exactly five open short-answer questions",
                      generation.QUIZ_PROMPT)
        self.assertIn("Do not write multiple choice questions",
                      generation.QUIZ_PROMPT)
        for question_type in ("definition", "explanation", "comparison",
                              "application"):
            self.assertIn(question_type, generation.QUIZ_PROMPT)


if __name__ == "__main__":
    unittest.main()
