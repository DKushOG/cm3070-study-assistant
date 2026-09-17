"""Pipeline controller.

Runs the full generation workflow for one source combination: build the
labelled context, generate revision notes, then generate the quiz from the
context plus those notes. Timing is recorded per step so the evaluation can
report processing time. The client is injected, which keeps this module
fully testable with a fake client and lets the real client point at any
OpenAI compatible endpoint.

The result also carries the reproducibility settings (model, endpoint,
temperature, seed), any context truncation, which extractor produced the
slide text, and the quiz format validation verdict, so save_result can write
a self-documenting header for the report appendices and nothing is lost
silently.

Quiz validation always runs; retry does not. quiz_max_attempts defaults to 1,
meaning the quiz is generated exactly once, so the default path makes the same
two model calls with the same prompts as the evaluated prototype. That keeps
baseline results valid while still measuring how often the documented format
failures occur.
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
    # Run metadata, defaulted so a fake client without these attributes still
    # produces a valid result. Populated from the injected client in
    # run_pipeline and recorded by save_result.
    model: str = None
    base_url: str = None
    temperature: float = None
    seed: int = None
    max_context_words: int = None
    truncation: list = field(default_factory=list)
    # Which extraction path produced the slide text, carried through from the
    # interface so evaluation tables can compare runs fed by Tesseract against
    # runs fed by the vision model. None records as "unset".
    slide_source: str = None
    # Which Whisper size produced the transcript, so a saved run can be matched
    # to the right arm of the word error rate comparison. None records as
    # "unset", e.g. when the transcript was pasted in by hand.
    transcript_source: str = None
    # Quiz format validation, recorded on every run even when retry is off.
    quiz_attempts: int = 1
    quiz_validation: object = None
    # The seed each quiz attempt was issued with, in order, so a run can be
    # reproduced attempt by attempt. Empty when no seed is configured.
    quiz_seeds: list = field(default_factory=list)
    # Set when retries provably cannot differ, so wasted attempts are reported
    # rather than silently burned.
    quiz_retry_note: str = None
    # Which generation path produced the quiz: the prompt-and-check path or
    # the schema-constrained path. Recorded on every run so a saved output can
    # always be attributed to one path, which is what makes the before-and-
    # after comparison in the evaluation defensible.
    quiz_structured: bool = False

    @property
    def total_seconds(self):
        return self.notes_seconds + self.quiz_seconds


def seed_for_attempt(base_seed, attempt):
    """Return the sampling seed for one quiz attempt.

    The rule is deterministic and documented so the whole retry sequence stays
    reproducible: attempt 1 uses the configured seed exactly, and attempt n
    uses base_seed + (n - 1). A single-attempt run is therefore unchanged, and
    a three-attempt run with LLM_SEED=42 uses 42, 43, 44 every time it is
    repeated.

    What the rule buys is reproducibility of a retry sequence while sampling
    is switched on: with LLM_TEMPERATURE above 0 each attempt samples
    differently, and pinning the seeds to a known series means the same run
    can be replayed attempt by attempt rather than only the first call being
    repeatable.

    It has no effect at LLM_TEMPERATURE=0. Decoding there is greedy, taking
    the highest-probability token at every step with no sampling for a seed to
    influence, so attempts carrying seeds 42, 43 and 44 return
    character-for-character identical text. Retry is futile at that
    temperature whatever the seed, which is the case _retry_determinism_note
    reports.
    """
    if base_seed is None:
        return None
    return base_seed + (attempt - 1)


def _client_for_attempt(client, attempt):
    """Return the client to issue one quiz attempt with.

    Attempt 1, and any run with no seed configured, uses the injected client
    unchanged, so the default path is byte-identical to before. A later
    attempt gets a shallow copy carrying the derived seed: the copy shares the
    cached connection, and the injected client is never mutated. The prompt is
    untouched either way, because the seed lives in client state and is read
    per request rather than being part of the prompt.
    """
    base_seed = getattr(client, "seed", None)
    if attempt == 1 or base_seed is None:
        return client, base_seed
    attempt_client = copy.copy(client)
    attempt_client.seed = seed_for_attempt(base_seed, attempt)
    return attempt_client, attempt_client.seed


def _retry_determinism_note(client, quiz_max_attempts):
    """Explain when configured retries cannot produce a different result.

    Temperature 0 is greedy decoding: the highest-probability token is taken
    at every step and no sampling takes place, so there is nothing for a seed
    to influence. Attempts issued with seeds 42, 43 and 44 return
    character-for-character identical text. Temperature is therefore checked
    **before** the seed and reported whether or not a seed is set: an earlier
    version returned as soon as a seed was present, which meant setting a seed
    silenced this warning while leaving the retries just as futile. That is the
    same failure shape as the two mistakes recorded in
    docs/prompt5_plan.md, a signal that stops reporting a problem without
    fixing it, so the order here is deliberate.

    Rather than silently burning model calls, the run records why the retries
    were futile. The attempts are still made as configured: short-circuiting
    them would change how many calls a configured retry issues, on an
    assumption about endpoint determinism this project has not measured.
    """
    if quiz_max_attempts <= 1:
        return None
    if getattr(client, "temperature", None) == 0:
        return ("retries cannot differ: temperature is 0, so decoding is "
                "greedy and no sampling takes place. LLM_SEED has no effect "
                "at this temperature, so varying it across attempts cannot "
                "change the output. Set LLM_TEMPERATURE above 0 for retry to "
                "be a real mechanism.")
    return None


def run_pipeline(client, transcript=None, slide_text=None, notes=None,
                 max_words=None, slide_source=None, transcript_source=None,
                 quiz_max_attempts=1, quiz_structured=None):
    """Run the two-call pipeline once and validate the resulting quiz.

    quiz_max_attempts defaults to 1: the quiz is generated exactly once and
    the run makes the same two model calls, with the same prompts, as before
    validation existed. When it is higher, a quiz that fails validation is
    regenerated using the identical unchanged prompt. If every attempt fails,
    the last attempt is kept, which is what "regenerate" plainly means and
    avoids inventing a ranking heuristic between two bad outputs.

    quiz_structured selects the generation path. None means "take the
    configured default", which is off, so the legacy prompt-and-check path
    remains the reference condition. Passing True issues the quiz call under a
    JSON schema instead. Both paths produce the same four-line text and are
    validated by the same unchanged validator.
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

    # quiz_seconds covers every attempt, so reported timing reflects the real
    # cost of producing the quiz that was kept.
    attempts = 0
    quiz = ""
    validation = None
    seeds = []
    for attempt in range(1, max(1, quiz_max_attempts) + 1):
        attempts = attempt
        # Only the seed varies between attempts; the prompt bytes are
        # identical because the same context and notes are passed every time.
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
        # Read defensively: the fake client used in tests need not expose
        # these, in which case they stay None ("unset" in the header).
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
    """Render a run setting for the header, distinguishing an unset value
    (None) from a real one such as 0, which is a meaningful temperature."""
    return "unset" if value is None else str(value)


def save_result(result, output_dir="outputs"):
    """Write a run to a timestamped text file and return the path.

    The header records the model, endpoint and reproducibility settings
    (temperature and seed), any context truncation, the extractor that
    produced the slide text and the quiz validation verdict, so every saved
    run is self-documenting for the report appendices and nothing is lost
    silently.
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
        # New fields are appended after the existing header lines so nothing
        # already read by the evaluation workbook shifts position.
        handle.write("quiz_seeds="
                     + (",".join(str(seed) for seed in result.quiz_seeds)
                        if result.quiz_seeds else "unset") + "\n")
        if result.quiz_retry_note:
            handle.write(f"quiz_retry_note={result.quiz_retry_note}\n")
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
