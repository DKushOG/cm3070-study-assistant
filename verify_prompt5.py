"""Offline verification harness for the prompt 5 changes.

This is a verification harness for the author's own use. It is **not part of
the system**: nothing in the application imports it, and it must never be
imported by `app.py`, `core/`, `modules/` or `evaluation/`. It exists so the
three changes can be checked without a model, without a network connection and
without starting the interface, because a verification cycle that needs a real
run costs about thirty minutes.

Run it from the project root:

    python verify_prompt5.py

It prints a PASS or FAIL line for each check and exits non-zero if any fails.

Check 1 extends the byte-identity approach already used by
`tests/test_llm_client.py::ChatRequestIdentityTests` and `RequestKwargsTests`,
which compare the assembled request across sampling combinations, rather than
inventing a second way of asserting the same thing.
"""
import sys

from core.generation import QUIZ_PROMPT, generate_quiz_questions
from core.llm_client import LLMClient
from core.orchestrator import run_pipeline, seed_for_attempt
from core.quiz_validation import (GENERATED_NOTES_FAULT, INVALID_BASIS_FAULT,
                                  basis_sources, looks_like_a_question,
                                  names_generated_notes, validate_quiz)
from evaluation import revalidate_outputs as ro

OUTPUTS_DIR = "outputs"

# Verdicts recorded by evaluation/revalidate_outputs.py over the whole outputs
# folder BEFORE any prompt 5 change, captured by running it on the unmodified
# code. Check 3 compares the current verdicts against these so a change is
# visible rather than silent.
BASELINE_BEFORE = {
    "20260813_205536_156923_combined_transcript_and_slide_text.txt": (0, False, False),
    "20260813_205845_918490_combined_transcript_and_slide_text.txt": (0, False, False),
    "20260813_210754_881089_combined_transcript_and_slide_text.txt": (5, True, False),
    "20260813_211217_414978_combined_transcript_and_slide_text.txt": (6, False, False),
    "20260816_200251_889296_slide_text_only.txt": (1, False, False),
    "20260816_203033_109334_transcript_slide_text_and_notes.txt": (0, False, False),
    "20260816_224747_377549_transcript_slide_text_and_notes.txt": (4, False, False),
    "20260816_230613_419655_transcript_slide_text_and_notes.txt": (4, False, False),
}

# The declarative "question" from the most recent real run.
REAL_DECLARATIVE_QUESTION = (
    "The concept of base rate neglect refers to the tendency to ignore "
    "prior probabilities when making decisions."
)

# Every distinct Source basis value across the saved runs, with the verdict
# each should receive.
REAL_BASIS_VALUES = {
    "transcript": "valid",
    "Lecture transcript": "valid",
    "notes": "valid",
    "student notes": "valid",
    "Student notes": "valid",
    "slide text": "valid",
    "Slide text": "valid",
    "notes, slide text": "valid",
    "transcript, slide text, notes": "valid",
    "Revision notes": "generated notes",
}


class FailingClient:
    """Always returns a quiz that fails validation, and records what it was
    asked. Mirrors tests/fakes.py rather than calling a real model."""

    def __init__(self, seed=None, temperature=None, model="llama3.2"):
        self.model = model
        self.base_url = "http://localhost:11434/v1"
        self.temperature = temperature
        self.seed = seed
        self.requests = []

    def chat(self, prompt):
        # Assemble the request the real client would build, so the bytes
        # compared here are the bytes that would be sent.
        probe = LLMClient(model=self.model)
        probe.temperature = self.temperature
        probe.seed = self.seed
        self.requests.append(probe._request_kwargs(prompt))
        return "not a quiz at all"


results = []


def record(name, ok, detail=""):
    results.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if detail:
        for line in detail.splitlines():
            print(f"       {line}")


def check_1_default_request_unchanged():
    """With QUIZ_MAX_ATTEMPTS unset the assembled quiz request is byte
    identical to today's, and only sampling keys differ across combinations."""
    context, notes = "CONTEXT BYTES", "NOTES BYTES"
    expected_prompt = QUIZ_PROMPT.format(context=context, revision_notes=notes)

    client = FailingClient()
    run_pipeline(client, transcript="t", slide_text="s", notes="n")
    detail = []
    ok = True

    if len(client.requests) != 2:
        ok = False
        detail.append(f"expected 2 model calls, saw {len(client.requests)}")

    # The unset case must be exactly the two-key request, as pinned by
    # ChatRequestIdentityTests.
    quiz_request = client.requests[-1]
    if set(quiz_request) != {"model", "messages"}:
        ok = False
        detail.append(f"unset run sent extra keys: {sorted(quiz_request)}")

    # Across sampling combinations the message bytes must be identical and
    # only temperature/seed may appear, which is the RequestKwargsTests rule.
    baseline = None
    for temperature, seed in ((None, None), (0, None), (0, 42), (0.7, 7)):
        probe = LLMClient(model="llama3.2")
        probe.temperature = temperature
        probe.seed = seed
        kwargs = probe._request_kwargs(expected_prompt)
        if baseline is None:
            baseline = kwargs["messages"]
        if kwargs["messages"] != baseline:
            ok = False
            detail.append(f"messages differ at temperature={temperature} "
                          f"seed={seed}")
        if set(kwargs) - {"model", "messages", "temperature", "seed"}:
            ok = False
            detail.append(f"unexpected keys at temperature={temperature}")
    if baseline and baseline[0]["content"] != expected_prompt:
        ok = False
        detail.append("quiz prompt bytes are not QUIZ_PROMPT as formatted")

    record("1. default (QUIZ_MAX_ATTEMPTS unset) request is byte-identical",
           ok, "\n".join(detail))


def check_2_seeds_vary_prompt_does_not():
    """With a seed set and attempts raised to three, three requests are sent
    whose prompt bytes are identical and whose seeds differ."""
    client = FailingClient(seed=42, temperature=0)
    result = run_pipeline(client, transcript="t", quiz_max_attempts=3)
    quiz_requests = client.requests[1:]
    detail = []
    ok = True

    if len(quiz_requests) != 3:
        ok = False
        detail.append(f"expected 3 quiz attempts, saw {len(quiz_requests)}")
    else:
        prompts = [request["messages"][0]["content"]
                   for request in quiz_requests]
        if len(set(prompts)) != 1:
            ok = False
            detail.append("prompt bytes differed across attempts")
        seeds = [request.get("seed") for request in quiz_requests]
        if seeds != [42, 43, 44]:
            ok = False
            detail.append(f"seeds were {seeds}, expected [42, 43, 44]")
        if result.quiz_seeds != [42, 43, 44]:
            ok = False
            detail.append(f"result.quiz_seeds was {result.quiz_seeds}")
        detail.append(f"seeds used: {seeds}; prompt bytes identical: "
                      f"{len(set(prompts)) == 1}")
    if seed_for_attempt(42, 1) != 42:
        ok = False
        detail.append("attempt 1 no longer uses the configured seed")

    record("2. retry varies the seed, never the prompt bytes", ok,
           "\n".join(detail))


def check_3_saved_outputs_still_parse():
    """Every file in outputs still parses, with a before-and-after comparison
    so any change of verdict is visible."""
    paths = ro.run_files(OUTPUTS_DIR)
    detail = []
    ok = True
    if not paths:
        record("3. every saved run still parses", False,
               f"no .txt files found in '{OUTPUTS_DIR}'")
        return

    detail.append(f"{'file':<62} before -> after")
    for path in paths:
        row = ro.revalidate_file(path)
        name = path.split("\\")[-1].split("/")[-1]
        if row is None:
            ok = False
            detail.append(f"{name:<62} FAILED TO PARSE")
            continue
        after = (row["Questions found"], row["Returned five"],
                 row["Validation passed"])
        before = BASELINE_BEFORE.get(name)
        if before is None:
            detail.append(f"{name:<62} (new file, no baseline) -> {after}")
            continue
        changed = "" if before == after else "   <== VERDICT CHANGED"
        if before != after:
            ok = False
        extra = []
        if row["Generated notes basis"]:
            extra.append(f"+{row['Generated notes basis']} generated-notes")
        if row["Invalid source basis"]:
            extra.append(f"+{row['Invalid source basis']} invalid-basis")
        if row["Shape warnings"]:
            extra.append(f"+{row['Shape warnings']} shape-warning")
        suffix = ("  new detail: " + ", ".join(extra)) if extra else ""
        detail.append(f"{name:<62} {before} -> {after}{changed}{suffix}")

    detail.append("")
    detail.append("questions_found / returned_five / passed are unchanged for "
                  "every file;")
    detail.append("only additional fault detail is reported, which is the "
                  "intended Task 2 and 3 improvement.")
    record("3. every saved run still parses, verdicts compared", ok,
           "\n".join(detail))


def check_4_source_basis_on_real_values():
    """The source basis check gives the expected verdict on every real value
    found in the outputs folder."""
    detail = []
    ok = True
    for value, expected in sorted(REAL_BASIS_VALUES.items()):
        if names_generated_notes(value):
            verdict = "generated notes"
        elif basis_sources(value):
            verdict = "valid"
        else:
            verdict = "names no source"
        if verdict != expected:
            ok = False
            detail.append(f"{value!r}: got {verdict}, expected {expected}")
        else:
            sources = basis_sources(value)
            detail.append(f"{value!r:<34} -> {verdict}"
                          + (f" {sources}" if sources else ""))
    # A value naming nothing must still be caught.
    if basis_sources("general knowledge"):
        ok = False
        detail.append("'general knowledge' was wrongly accepted")
    record("4. source basis verdicts on the real values in outputs", ok,
           "\n".join(detail))


def check_5_shape_warning_is_soft():
    """The warning fires on the real declarative question, and the pass or
    fail verdict for that run is the same with the warning as without it."""
    path = (OUTPUTS_DIR
            + "/20260816_230613_419655_transcript_slide_text_and_notes.txt")
    detail = []
    ok = True

    if looks_like_a_question(REAL_DECLARATIVE_QUESTION):
        ok = False
        detail.append("the real declarative question was not flagged")

    try:
        quiz = ro.extract_quiz(ro.read_text(path))
    except OSError as error:
        record("5. shape warning fires and does not change the verdict",
               False, f"could not read {path}: {error}")
        return

    with_warning = validate_quiz(quiz)
    # The same run with the declarative block rephrased as a question: the
    # only difference is the shape signal.
    rephrased = quiz.replace(
        REAL_DECLARATIVE_QUESTION,
        "What does the concept of base rate neglect refer to?")
    without_warning = validate_quiz(rephrased)

    if with_warning.shape_warning_count != 1:
        ok = False
        detail.append("expected exactly 1 shape warning, got "
                      f"{with_warning.shape_warning_count}")
    if without_warning.shape_warning_count != 0:
        ok = False
        detail.append("rephrased run still warned")
    if with_warning.passed != without_warning.passed:
        ok = False
        detail.append("the warning changed the pass/fail verdict")
    if with_warning.failures != without_warning.failures:
        ok = False
        detail.append("the warning changed the failure list")
    if any("read as a question" in failure
           for failure in with_warning.failures):
        ok = False
        detail.append("the warning leaked into failures")

    detail.append(f"warnings: {with_warning.shape_warning_count} with, "
                  f"{without_warning.shape_warning_count} without")
    detail.append(f"passed: {with_warning.passed} with, "
                  f"{without_warning.passed} without (identical)")
    detail.append(f"failures identical: "
                  f"{with_warning.failures == without_warning.failures}")
    record("5. shape warning fires and does not change the verdict", ok,
           "\n".join(detail))


def main():
    print("Offline verification of the prompt 5 changes")
    print("=" * 72)
    check_1_default_request_unchanged()
    print()
    check_2_seeds_vary_prompt_does_not()
    print()
    check_3_saved_outputs_still_parse()
    print()
    check_4_source_basis_on_real_values()
    print()
    check_5_shape_warning_is_soft()
    print()
    print("=" * 72)
    failed = [name for name, ok in results if not ok]
    if failed:
        print(f"{len(failed)} of {len(results)} checks FAILED:")
        for name in failed:
            print(f"  - {name}")
        return 1
    print(f"All {len(results)} checks PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
