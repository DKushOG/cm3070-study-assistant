"""Pipeline controller.

Runs one job from start to finish: build the labelled context, generate the
revision notes, then generate the quiz from the context and those notes.

The model client is passed in rather than created here, so tests can inject a
fake one. Processing times and run settings are recorded so the saved output
can be checked later.
"""
import copy
import os
import time
from dataclasses import dataclass, field
from datetime import datetime

import config
from core.context_builder import build_context_with_report, mode_name
from core.generation import (generate_revision_notes,
                             generate_quiz_questions,
                             generate_quiz_structured)
from core.quiz_validation import validate_quiz


@dataclass
class PipelineResult:
    mode: str
    context: str
    notes: str
    quiz: str
    notes_seconds: float
    quiz_seconds: float
    # Run settings, read from the client in run_pipeline. They default to None
    # so a fake client in tests still produces a valid result.
    model: str = None
    base_url: str = None
    temperature: float = None
    seed: int = None
    max_context_words: int = None
    truncation: list = field(default_factory=list)
    # Which extractor produced the slide text. None is recorded as "unset".
    slide_source: str = None
    # Which Whisper size produced the transcript. None is recorded as "unset".
    transcript_source: str = None
    # Quiz format validation, recorded on every run even when retry is off.
    quiz_attempts: int = 1
    quiz_validation: object = None
    # Seed used for each quiz attempt, in order, so a run can be repeated.
    quiz_seeds: list = field(default_factory=list)
    # Set when a retry is unlikely to change the output, so the waste shows.
    quiz_retry_note: str = None
    # Which quiz path ran, the prompt-and-check path or the schema path.
    quiz_structured: bool = False

    @property
    def total_seconds(self):
        return self.notes_seconds + self.quiz_seconds


def seed_for_attempt(base_seed, attempt):
    """Return the sampling seed for one quiz attempt.

    Attempt 1 uses the configured seed and attempt n uses seed + (n - 1), so a
    retry sequence can be repeated exactly. Returns None when no seed is set.
    """
    if base_seed is None:
        return None
    return base_seed + (attempt - 1)


def _client_for_attempt(client, attempt):
    """Return the client to use for one quiz attempt.

    Attempt 1 uses the injected client unchanged. A later attempt gets a
    shallow copy carrying the derived seed, so the injected client is never
    modified and the prompt stays the same.
    """
    base_seed = getattr(client, "seed", None)
    if attempt == 1 or base_seed is None:
        return client, base_seed
    attempt_client = copy.copy(client)
    attempt_client.seed = seed_for_attempt(base_seed, attempt)
    return attempt_client, attempt_client.seed


def _retry_determinism_note(client, quiz_max_attempts):
    """Explain when retrying is unlikely to change the quiz.

    At temperature 0, changing the seed is not expected to produce a different
    result because sampling is disabled.

    Returns a short explanation for the saved output, or None when retries may
    produce a different result.
    """
    if quiz_max_attempts <= 1:
        return None
    if getattr(client, "temperature", None) == 0:
        return ("retries are unlikely to differ: temperature is 0, so "
                "decoding is greedy and sampling is disabled. Changing "
                "LLM_SEED is not expected to change the output at this "
                "temperature. Set LLM_TEMPERATURE above 0 for retry to be a "
                "real mechanism.")
    return None


def run_pipeline(client, transcript=None, slide_text=None, notes=None,
                 max_words=None, slide_source=None, transcript_source=None,
                 quiz_max_attempts=1, quiz_structured=None):
    """Run the two-call pipeline once and validate the quiz.

    quiz_max_attempts defaults to 1, so the default run makes the same two
    model calls as before validation existed. A higher value regenerates a
    failing quiz with the same prompt and keeps the last attempt if all of
    them fail. quiz_structured picks the quiz path, and None means use the
    configured default.
    """
    if quiz_structured is None:
        quiz_structured = config.QUIZ_STRUCTURED

    context, truncation = build_context_with_report(
        transcript=transcript, slide_text=slide_text, notes=notes,
        max_words=max_words)
    mode = mode_name(transcript=transcript, slide_text=slide_text, notes=notes)

    start = time.perf_counter()
    revision_notes = generate_revision_notes(client, context)
    mid = time.perf_counter()

    # Include the time spent on every quiz attempt, including retries.
    attempts = 0
    quiz = ""
    validation = None
    seeds = []
    for attempt in range(1, max(1, quiz_max_attempts) + 1):
        attempts = attempt
        # Only the seed changes between attempts. The prompt stays identical.
        attempt_client, attempt_seed = _client_for_attempt(client, attempt)
        if attempt_seed is not None:
            seeds.append(attempt_seed)
        if quiz_structured:
            quiz = generate_quiz_structured(attempt_client, context,
                                            revision_notes)
        else:
            quiz = generate_quiz_questions(attempt_client, context,
                                           revision_notes)
        validation = validate_quiz(quiz)
        if validation.passed:
            break
    end = time.perf_counter()

    return PipelineResult(
        mode=mode,
        context=context,
        notes=revision_notes,
        quiz=quiz,
        notes_seconds=mid - start,
        quiz_seconds=end - mid,
        # Read defensively: a fake client in tests need not have these.
        model=getattr(client, "model", None),
        base_url=getattr(client, "base_url", None),
        temperature=getattr(client, "temperature", None),
        seed=getattr(client, "seed", None),
        max_context_words=max_words,
        truncation=truncation,
        slide_source=slide_source,
        transcript_source=transcript_source,
        quiz_attempts=attempts,
        quiz_validation=validation,
        quiz_seeds=seeds,
        quiz_retry_note=_retry_determinism_note(client, quiz_max_attempts),
        quiz_structured=quiz_structured,
    )


def _format_setting(value):
    """Render one setting for the header, keeping 0 different from unset."""
    return "unset" if value is None else str(value)


def save_result(result, output_dir="outputs"):
    """Write the run to a timestamped text file and return the path.

    The header records the model, endpoint, sampling settings, any truncation,
    the slide source and the quiz verdict, so a saved run can be read on its
    own later.
    """
    os.makedirs(output_dir, exist_ok=True)
    safe_mode = result.mode.lower().replace(" ", "_").replace(",", "")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = stamp + "_" + safe_mode + ".txt"
    path = os.path.join(output_dir, filename)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"MODE: {result.mode}\n")
        handle.write(f"model={_format_setting(result.model)}\n")
        handle.write(f"base_url={_format_setting(result.base_url)}\n")
        handle.write(f"temperature={_format_setting(result.temperature)}\n")
        handle.write(f"seed={_format_setting(result.seed)}\n")
        handle.write(
            f"max_context_words={_format_setting(result.max_context_words)}\n")
        handle.write(f"slide_source={_format_setting(result.slide_source)}\n")
        handle.write(
            f"transcript_source={_format_setting(result.transcript_source)}\n")
        handle.write(f"quiz_attempts={result.quiz_attempts}\n")
        validation = result.quiz_validation
        if validation is None:
            handle.write("quiz_validation=unset\n")
        else:
            handle.write(
                f"quiz_validation={'passed' if validation.passed else 'failed'}\n")
            handle.write(f"questions_found={validation.questions_found}\n")
            handle.write(f"returned_five={validation.returned_five}\n")
            if validation.failures:
                handle.write("quiz_validation_failures:\n")
                for failure in validation.failures:
                    handle.write(f"  {failure}\n")
        if result.truncation:
            handle.write("truncation:\n")
            for event in result.truncation:
                handle.write(f"  {event.describe()}\n")
        else:
            handle.write("truncation: none\n")
        # Added after the existing header lines, so older files still line up.
        handle.write("quiz_seeds="
                     + (",".join(str(seed) for seed in result.quiz_seeds)
                        if result.quiz_seeds else "unset") + "\n")
        if result.quiz_retry_note:
            handle.write(f"quiz_retry_note={result.quiz_retry_note}\n")
        handle.write(f"quiz_structured={bool(result.quiz_structured)}\n")
        if validation is not None:
            handle.write(
                f"quiz_shape_warnings={validation.shape_warning_count}\n")
            if validation.warnings:
                handle.write("quiz_warnings:\n")
                for warning in validation.warnings:
                    handle.write(f"  {warning}\n")
        handle.write("\n")
        handle.write("=== REVISION NOTES ===\n")
        handle.write(result.notes + "\n\n")
        handle.write("=== QUIZ QUESTIONS ===\n")
        handle.write(result.quiz + "\n\n")
        handle.write(f"notes_seconds={result.notes_seconds:.2f}\n")
        handle.write(f"quiz_seconds={result.quiz_seconds:.2f}\n")
        handle.write(f"total_seconds={result.total_seconds:.2f}\n")
    return path
