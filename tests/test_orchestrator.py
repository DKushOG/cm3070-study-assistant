import os
import tempfile
import unittest

from core.context_builder import TRANSCRIPT_LABEL
from core.orchestrator import run_pipeline, save_result, seed_for_attempt
from tests.fakes import FakeLLMClient


def read(path):
    with open(path, encoding="utf-8") as handle:
        return handle.read()


class RunPipelineTests(unittest.TestCase):
    def test_result_fields_populated(self):
        client = FakeLLMClient(replies=["NOTES OUT", "QUIZ OUT"])
        result = run_pipeline(client, transcript="t", slide_text="s")
        self.assertEqual(result.mode, "Combined transcript and slide text")
        self.assertEqual(result.notes, "NOTES OUT")
        self.assertEqual(result.quiz, "QUIZ OUT")
        self.assertIn("[Lecture transcript]", result.context)
        self.assertGreaterEqual(result.notes_seconds, 0)
        self.assertGreaterEqual(result.quiz_seconds, 0)
        self.assertAlmostEqual(
            result.total_seconds,
            result.notes_seconds + result.quiz_seconds)

    def test_quiz_call_depends_on_generated_notes(self):
        """The second model call must receive the notes produced by the
        first call. This is the sequential orchestration shown in Figure
        3.2 of the report."""
        client = FakeLLMClient(replies=["GENERATED NOTES XYZ", "QUIZ"])
        run_pipeline(client, transcript="t")
        self.assertEqual(len(client.prompts), 2)
        self.assertNotIn("GENERATED NOTES XYZ", client.prompts[0])
        self.assertIn("GENERATED NOTES XYZ", client.prompts[1])

    def test_no_sources_raises(self):
        with self.assertRaises(ValueError):
            run_pipeline(FakeLLMClient())


class SaveResultTests(unittest.TestCase):
    def test_writes_output_file(self):
        client = FakeLLMClient(replies=["N", "Q"])
        result = run_pipeline(client, notes="some notes")
        with tempfile.TemporaryDirectory() as tmp:
            path = save_result(result, output_dir=tmp)
            self.assertTrue(os.path.exists(path))
            with open(path, encoding="utf-8") as handle:
                content = handle.read()
            self.assertIn("MODE: Notes only", content)
            self.assertIn("=== REVISION NOTES ===", content)
            self.assertIn("=== QUIZ QUESTIONS ===", content)

    def test_rapid_saves_do_not_collide(self):
        """Two saves of the same mode in the same second must produce two
        files, not silently overwrite one, since an evaluation campaign
        depends on every output being kept."""
        client = FakeLLMClient(replies=["N", "Q"])
        result = run_pipeline(client, notes="some notes")
        with tempfile.TemporaryDirectory() as tmp:
            first = save_result(result, output_dir=tmp)
            second = save_result(result, output_dir=tmp)
            self.assertNotEqual(first, second)
            self.assertEqual(len(os.listdir(tmp)), 2)


class ResultMetadataTests(unittest.TestCase):
    def test_settings_captured_from_client(self):
        client = FakeLLMClient(replies=["N", "Q"], model="m", base_url="u",
                               temperature=0, seed=7)
        result = run_pipeline(client, notes="hi")
        self.assertEqual(result.model, "m")
        self.assertEqual(result.base_url, "u")
        self.assertEqual(result.temperature, 0)
        self.assertEqual(result.seed, 7)

    def test_settings_default_to_none_for_a_bare_client(self):
        result = run_pipeline(FakeLLMClient(), notes="hi")
        self.assertIsNone(result.model)
        self.assertIsNone(result.temperature)
        self.assertEqual(result.truncation, [])


class SaveResultHeaderTests(unittest.TestCase):
    def test_header_records_settings_and_truncation(self):
        client = FakeLLMClient(replies=["N", "Q"], model="m", base_url="u",
                               temperature=0, seed=7)
        result = run_pipeline(
            client, transcript="one two three four five", max_words=2)
        with tempfile.TemporaryDirectory() as tmp:
            content = read(save_result(result, output_dir=tmp))
        self.assertIn("model=m", content)
        self.assertIn("base_url=u", content)
        self.assertIn("temperature=0", content)
        self.assertIn("seed=7", content)
        self.assertIn("max_context_words=2", content)
        self.assertIn("truncation:", content)
        # The affected source is named and the dropped count is recorded, so
        # the run appendix shows exactly what was lost.
        self.assertIn(TRANSCRIPT_LABEL, content)
        self.assertIn("3 dropped", content)

    def test_header_reports_unset_settings_and_no_truncation(self):
        result = run_pipeline(FakeLLMClient(replies=["N", "Q"]), notes="a b c")
        with tempfile.TemporaryDirectory() as tmp:
            content = read(save_result(result, output_dir=tmp))
        self.assertIn("temperature=unset", content)
        self.assertIn("seed=unset", content)
        self.assertIn("max_context_words=unset", content)
        self.assertIn("truncation: none", content)

    def test_header_records_which_quiz_path_ran(self):
        """The interface can switch between the two quiz paths, so a saved
        run has to say which one produced its quiz."""
        result = run_pipeline(FakeLLMClient(replies=["N", "Q"]), notes="a b c")
        with tempfile.TemporaryDirectory() as tmp:
            content = read(save_result(result, output_dir=tmp))
        self.assertIn("quiz_structured=False", content)


GOOD_QUIZ = "\n".join(
    f"Question: Q{n}?\nSuggested answer: A{n}.\n"
    f"Question type: definition\nSource basis: transcript\n"
    for n in range(1, 6))
BAD_QUIZ = "Question: Only one?\nSuggested answer: A.\n" \
           "Question type: definition\nSource basis: transcript\n"


class QuizValidationRecordingTests(unittest.TestCase):
    def test_validation_recorded_on_default_settings(self):
        """Validation runs even with retry disabled, so the failure rate is
        measured on the default configuration."""
        client = FakeLLMClient(replies=["NOTES", BAD_QUIZ])
        result = run_pipeline(client, transcript="t")
        self.assertEqual(result.quiz_attempts, 1)
        self.assertFalse(result.quiz_validation.passed)
        self.assertEqual(result.quiz_validation.questions_found, 1)

    def test_default_makes_exactly_two_model_calls(self):
        """The default path must be byte-identical to the behaviour before
        validation existed: one notes call, one quiz call, no retry."""
        client = FakeLLMClient(replies=["NOTES", BAD_QUIZ])
        run_pipeline(client, transcript="t")
        self.assertEqual(len(client.prompts), 2)

    def test_passing_quiz_is_recorded_as_passed(self):
        client = FakeLLMClient(replies=["NOTES", GOOD_QUIZ])
        result = run_pipeline(client, transcript="t")
        self.assertTrue(result.quiz_validation.passed)
        self.assertTrue(result.quiz_validation.returned_five)


class QuizRetryTests(unittest.TestCase):
    def test_retry_regenerates_until_valid(self):
        client = FakeLLMClient(replies=["NOTES", BAD_QUIZ, GOOD_QUIZ])
        result = run_pipeline(client, transcript="t", quiz_max_attempts=3)
        self.assertEqual(result.quiz_attempts, 2)
        self.assertTrue(result.quiz_validation.passed)
        self.assertEqual(result.quiz, GOOD_QUIZ)

    def test_retry_reuses_an_identical_prompt(self):
        """The retry must not alter the evaluated prompt in any way."""
        client = FakeLLMClient(replies=["NOTES", BAD_QUIZ, BAD_QUIZ])
        run_pipeline(client, transcript="t", quiz_max_attempts=2)
        self.assertEqual(len(client.prompts), 3)
        self.assertEqual(client.prompts[1], client.prompts[2])

    def test_valid_first_attempt_does_not_retry(self):
        client = FakeLLMClient(replies=["NOTES", GOOD_QUIZ, GOOD_QUIZ])
        result = run_pipeline(client, transcript="t", quiz_max_attempts=3)
        self.assertEqual(result.quiz_attempts, 1)
        self.assertEqual(len(client.prompts), 2)

    def test_all_attempts_failing_keeps_the_last_and_reports_failure(self):
        client = FakeLLMClient(replies=["NOTES", BAD_QUIZ])
        result = run_pipeline(client, transcript="t", quiz_max_attempts=3)
        self.assertEqual(result.quiz_attempts, 3)
        self.assertFalse(result.quiz_validation.passed)
        self.assertEqual(result.quiz, BAD_QUIZ)


class RetrySeedTests(unittest.TestCase):
    """Retry regenerates with the identical prompt, so with a fixed seed and
    temperature 0 every attempt reproduced the same failure and the extra
    calls were wasted. Varying only the seed makes retry a real mechanism."""

    def test_seed_rule_is_deterministic_and_keeps_attempt_one(self):
        self.assertEqual(seed_for_attempt(42, 1), 42)
        self.assertEqual(seed_for_attempt(42, 2), 43)
        self.assertEqual(seed_for_attempt(42, 3), 44)
        self.assertIsNone(seed_for_attempt(None, 3))

    def test_prompt_bytes_identical_and_seeds_differ_across_attempts(self):
        client = FakeLLMClient(replies=["NOTES", BAD_QUIZ], seed=42)
        result = run_pipeline(client, transcript="t", quiz_max_attempts=3)
        # One notes call plus three quiz calls.
        self.assertEqual(len(client.prompts), 4)
        quiz_prompts = client.prompts[1:]
        self.assertEqual(quiz_prompts[0], quiz_prompts[1])
        self.assertEqual(quiz_prompts[1], quiz_prompts[2])
        self.assertEqual(client.seeds_used[1:], [42, 43, 44])
        self.assertEqual(result.quiz_seeds, [42, 43, 44])

    def test_injected_client_is_not_mutated(self):
        """Later attempts use a shallow copy, so the caller's client keeps the
        seed it was configured with."""
        client = FakeLLMClient(replies=["NOTES", BAD_QUIZ], seed=42)
        run_pipeline(client, transcript="t", quiz_max_attempts=3)
        self.assertEqual(client.seed, 42)

    def test_first_attempt_uses_the_configured_seed_exactly(self):
        client = FakeLLMClient(replies=["NOTES", GOOD_QUIZ], seed=7)
        result = run_pipeline(client, transcript="t", quiz_max_attempts=3)
        self.assertEqual(result.quiz_seeds, [7])
        self.assertEqual(client.seeds_used[1], 7)

    def test_no_seed_set_leaves_behaviour_unchanged(self):
        client = FakeLLMClient(replies=["NOTES", BAD_QUIZ])
        result = run_pipeline(client, transcript="t", quiz_max_attempts=3)
        self.assertEqual(len(client.prompts), 4)
        self.assertEqual(result.quiz_seeds, [])
        self.assertEqual(set(client.seeds_used), {None})

    def test_default_single_attempt_changes_nothing(self):
        client = FakeLLMClient(replies=["NOTES", BAD_QUIZ], seed=42)
        result = run_pipeline(client, transcript="t")
        self.assertEqual(len(client.prompts), 2)
        self.assertEqual(client.seeds_used, [42, 42])
        self.assertEqual(result.quiz_seeds, [42])
        self.assertIsNone(result.quiz_retry_note)

    def test_seeds_recorded_in_the_saved_header(self):
        client = FakeLLMClient(replies=["NOTES", BAD_QUIZ], seed=42)
        result = run_pipeline(client, transcript="t", quiz_max_attempts=3)
        with tempfile.TemporaryDirectory() as tmp:
            content = read(save_result(result, output_dir=tmp))
        self.assertIn("quiz_seeds=42,43,44", content)

    def test_no_seed_records_unset_in_the_header(self):
        result = run_pipeline(FakeLLMClient(replies=["NOTES", GOOD_QUIZ]),
                              transcript="t")
        with tempfile.TemporaryDirectory() as tmp:
            content = read(save_result(result, output_dir=tmp))
        self.assertIn("quiz_seeds=unset", content)


class RetryDeterminismNoteTests(unittest.TestCase):
    """Greedy decoding with no seed cannot be varied, so the run says so
    rather than silently burning attempts."""

    def test_note_set_when_temperature_zero_and_no_seed(self):
        client = FakeLLMClient(replies=["NOTES", BAD_QUIZ], temperature=0)
        result = run_pipeline(client, transcript="t", quiz_max_attempts=3)
        self.assertIsNotNone(result.quiz_retry_note)
        self.assertIn("LLM_SEED", result.quiz_retry_note)
        # The attempts are still made as configured.
        self.assertEqual(result.quiz_attempts, 3)

    def test_note_still_fires_when_a_seed_is_set_at_temperature_zero(self):
        """Regression: an earlier version returned as soon as a seed was
        present, so setting a seed silenced the warning without removing the
        futility. Temperature 0 is greedy decoding, so the seed has no effect
        and the attempts are still identical."""
        client = FakeLLMClient(replies=["NOTES", BAD_QUIZ], temperature=0,
                               seed=42)
        result = run_pipeline(client, transcript="t", quiz_max_attempts=3)
        self.assertIsNotNone(result.quiz_retry_note)
        # The reason given must be the temperature, not the seed.
        self.assertIn("temperature is 0", result.quiz_retry_note)
        self.assertIn("greedy", result.quiz_retry_note)
        self.assertIn("LLM_TEMPERATURE above 0", result.quiz_retry_note)
        # The seeds are still varied and recorded; they simply cannot help
        # at this temperature.
        self.assertEqual(result.quiz_seeds, [42, 43, 44])

    def test_no_note_when_sampling_can_actually_vary(self):
        """Above 0 the decoding samples, so a varied seed changes the output
        and retry is a real mechanism. Nothing to warn about."""
        client = FakeLLMClient(replies=["NOTES", BAD_QUIZ], temperature=0.7,
                               seed=42)
        result = run_pipeline(client, transcript="t", quiz_max_attempts=3)
        self.assertIsNone(result.quiz_retry_note)
        self.assertEqual(result.quiz_seeds, [42, 43, 44])

    def test_no_note_when_temperature_is_unset(self):
        """Unset means the model's own default, which samples."""
        client = FakeLLMClient(replies=["NOTES", BAD_QUIZ], seed=42)
        result = run_pipeline(client, transcript="t", quiz_max_attempts=3)
        self.assertIsNone(result.quiz_retry_note)

    def test_no_note_on_a_single_attempt_run(self):
        client = FakeLLMClient(replies=["NOTES", BAD_QUIZ], temperature=0)
        self.assertIsNone(run_pipeline(client, transcript="t").quiz_retry_note)

    def test_note_written_to_the_saved_header(self):
        client = FakeLLMClient(replies=["NOTES", BAD_QUIZ], temperature=0)
        result = run_pipeline(client, transcript="t", quiz_max_attempts=3)
        with tempfile.TemporaryDirectory() as tmp:
            content = read(save_result(result, output_dir=tmp))
        self.assertIn("quiz_retry_note=", content)


class SlideSourceTests(unittest.TestCase):
    def test_slide_source_recorded_on_result_and_in_header(self):
        client = FakeLLMClient(replies=["N", GOOD_QUIZ])
        result = run_pipeline(client, slide_text="s",
                              slide_source="Vision model (qwen3-vl:4b)")
        self.assertEqual(result.slide_source, "Vision model (qwen3-vl:4b)")
        with tempfile.TemporaryDirectory() as tmp:
            content = read(save_result(result, output_dir=tmp))
        self.assertIn("slide_source=Vision model (qwen3-vl:4b)", content)
        self.assertIn("quiz_attempts=1", content)
        self.assertIn("quiz_validation=passed", content)
        self.assertIn("returned_five=True", content)

    def test_transcript_source_records_the_whisper_size(self):
        """The saved run must say which Whisper size produced the transcript,
        so it can be matched to the right arm of the WER comparison."""
        client = FakeLLMClient(replies=["N", GOOD_QUIZ])
        result = run_pipeline(client, transcript="t",
                              transcript_source="Whisper small")
        self.assertEqual(result.transcript_source, "Whisper small")
        with tempfile.TemporaryDirectory() as tmp:
            content = read(save_result(result, output_dir=tmp))
        self.assertIn("transcript_source=Whisper small", content)

    def test_unknown_transcript_source_records_as_unset(self):
        result = run_pipeline(FakeLLMClient(replies=["N", GOOD_QUIZ]),
                              transcript="pasted by hand")
        with tempfile.TemporaryDirectory() as tmp:
            content = read(save_result(result, output_dir=tmp))
        self.assertIn("transcript_source=unset", content)

    def test_unknown_slide_source_records_as_unset(self):
        result = run_pipeline(FakeLLMClient(replies=["N", BAD_QUIZ]),
                              notes="n")
        with tempfile.TemporaryDirectory() as tmp:
            content = read(save_result(result, output_dir=tmp))
        self.assertIn("slide_source=unset", content)
        self.assertIn("quiz_validation=failed", content)
        self.assertIn("quiz_validation_failures:", content)


if __name__ == "__main__":
    unittest.main()
