# Prompt 5 plan — retry determinism, source basis, question shape

Written before any code was changed, from reading `README.md`,
`core/generation.py`, `core/quiz_validation.py`, `core/orchestrator.py`,
`core/llm_client.py`, `config.py`, `evaluation/revalidate_outputs.py`,
`evaluation/run_eval.py`, and all eight saved runs in `outputs`.

Precondition check requested in the prompt:
`outputs\20260816_230613_419655_transcript_slide_text_and_notes.txt` contains
**four** lines beginning `Question:` after the `=== QUIZ QUESTIONS ===` marker.
Four was the "proceed" case, so this plan proceeds.

---

## 1. Exact current behaviour of the retry loop

`core/orchestrator.py`, inside `run_pipeline`:

```python
    attempts = 0
    quiz = ""
    validation = None
    for attempt in range(1, max(1, quiz_max_attempts) + 1):
        attempts = attempt
        quiz = generate_quiz_questions(client, context, revision_notes)
        validation = validate_quiz(quiz)
        if validation.passed:
            break
```

Facts that follow from those lines:

- `quiz_max_attempts` defaults to `1` in the signature, so the default path
  runs the loop body exactly once and makes exactly one quiz call.
- Every attempt calls `generate_quiz_questions(client, context,
  revision_notes)` with the same three arguments, so the prompt bytes are
  already identical across attempts. Nothing needs to change to preserve that.
- The loop stops early only on `validation.passed`. If every attempt fails the
  last attempt is kept.
- Nothing between attempts alters the client, so **every attempt is issued
  with exactly the same sampling settings**.

## 2. Is the sampling seed reused across attempts?

**Yes.** The seed is client state, read fresh on each request but never
changed. `core/llm_client.py`:

```python
        self.seed = seed if seed is not None else config.LLM_SEED
```

```python
    def _apply_settings(self, kwargs):
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.seed is not None:
            kwargs["seed"] = self.seed
        return kwargs
```

`_apply_settings` reads `self.seed` on every call, and the retry loop never
assigns to it, so attempts 1..N all send the same `seed` value. With
`LLM_TEMPERATURE=0` and `LLM_SEED=42` the endpoint therefore receives an
identical request each time and returns identical text, so each retry
reproduces the same failure. The report's claim that retry recovers format
failures is false under exactly those settings. This confirms the defect.

**The seed can be varied per attempt**, so the fallback route in Task 1 ("if
the seed cannot be varied, detect and report instead") does not apply. Because
`_apply_settings` reads `self.seed` at call time, giving one attempt a
different seed requires only an object whose `seed` attribute differs. No
change to `core/generation.py` and no change to `LLMClient` is needed.

## 3. Source basis values found across the real saved runs

Every distinct value of `Source basis:` in `outputs`, verbatim:

| Value | File(s) | Verdict intended |
|---|---|---|
| `transcript` | 211217, 203033 (x2), 210754 (x2) | valid, transcript |
| `Lecture transcript` | 224747 (x2), 230613 (x2) | valid, transcript |
| `notes` | 211217 (x2), 203033 (x2), 210754 | valid, notes |
| `student notes` | 203033 | valid, notes |
| `Student notes` | 224747, 230613 | valid, notes |
| `slide text` | 210754 (x2) | valid, slide text |
| `Slide text` | 200251 | valid, slide text |
| `notes, slide text` | 203033 | valid, two sources |
| `transcript, slide text, notes` | 210754 | valid, three sources |
| `Revision notes` | 224747, 230613 | **fault: names the generated notes** |

`Revision notes` is the failure Task 2 describes: it attributes the answer to
the notes produced by the system's own first model call, not to any input the
student supplied.

Note the ambiguity this creates and the rule chosen for it: "notes" alone means
the student notes input, while "Revision notes" names the generated notes.
"Student notes" is unambiguous. The check therefore looks for the word
"revision" (and "generated") as the marker of the generated-notes fault, and
treats a bare "notes" as the student notes input, which is what the prompt
template asks for.

## 4. Fields to add, and where each surfaces

| Field | Meaning | Surfaces in |
|---|---|---|
| `PipelineResult.quiz_seeds` | Seed used by each quiz attempt, in order | saved header as `quiz_seeds=`, appended after the existing header lines |
| `PipelineResult.quiz_retry_note` | Set when retries provably cannot differ (temperature 0 with no seed), so wasted attempts are reported rather than silent | saved header as `quiz_retry_note=` |
| `QuestionBlock.basis_faults` / `INVALID_BASIS_FAULT`, `GENERATED_NOTES_FAULT` | Source basis names no input source / names the generated revision notes | existing `failures` list, so it flows into the saved header, `run_eval` CSV and `revalidate_outputs` CSV through the existing mechanism |
| `QuestionBlock.shape_warnings` / `QuizValidation.shape_warnings` | Soft signal: question text does not look like a question | new columns appended last, never part of `passed` |

Naming and ordering constraints observed: no existing field is renamed,
reordered or repurposed. New CSV columns are appended **after all existing
columns, including the rubric columns**, so existing column positions are
untouched. New saved-header lines are appended after `truncation:`, which is
currently the last header line.

## 5. Files to touch

| File | Why |
|---|---|
| `core/quiz_validation.py` | Task 2 source basis check, Task 3 shape warning |
| `core/orchestrator.py` | Task 1 per-attempt seed, new result fields, header lines |
| `evaluation/run_eval.py` | append new columns (Task 2 fault counts, Task 3 warnings) |
| `evaluation/revalidate_outputs.py` | append new columns |
| `tests/fakes.py` | record the seed in force at each call so tests can assert it |
| `tests/test_quiz_validation.py` | Task 2 and 3 tests, fixtures copied verbatim from real runs |
| `tests/test_orchestrator.py` | Task 1 tests |
| `tests/test_run_eval.py`, `tests/test_revalidate_outputs.py` | new columns |
| `verify_prompt5.py` (project root) | offline verification harness, not imported by the app |
| `docs/prompt5_plan.md` | this file |

Not touched: `core/generation.py` (no prompt text change, and no signature
change is needed), `core/llm_client.py`, `core/context_builder.py`, `app.py`.

## 6. Conflicts with the constraints

No task conflicts with a constraint, so nothing is being skipped. Three points
need stating rather than silently deciding:

1. **Existing saved runs will report more faults than before.** The source
   basis check is new, so `224747` and `230613` will now additionally report
   the `Revision notes` fault. This is the intended improvement Task 2 asks
   for, not an accident. It cannot flip any file from pass to fail, because a
   baseline run of `revalidate_outputs` over all eight files before any change
   reports **0 of 8 passing**; every file already failed on question count or
   labels. The pass rate is unchanged at 0 of 8.
2. **Temperature 0 is deterministic and varying the seed cannot help, with
   or without a seed set.** Temperature 0 is greedy decoding: the
   highest-probability token is taken at every step and no sampling occurs, so
   there is nothing for a seed to influence. Attempts carrying seeds 42, 43
   and 44 return character-for-character identical text. That is the
   configuration of the most recent run (`temperature=0.0`, `seed=unset`,
   `quiz_attempts=3`) and of the working `.env`.

   Rather than silently burning attempts, the run records `quiz_retry_note`
   stating that decoding is greedy, that `LLM_SEED` has no effect at this
   temperature, and that `LLM_TEMPERATURE` must be above 0 for retry to be a
   real mechanism. The loop is **not** stopped early: doing so would change
   how many model calls a configured retry makes, based on an assumption about
   the endpoint's determinism that this project has not measured. Reporting is
   honest; short-circuiting on an unmeasured assumption is not.

   **Correction — the original reasoning in this section was wrong.** As first
   written, this point claimed the futility applied only when *no* seed was
   set, and that setting `LLM_SEED` would make the retries vary. That was a
   misunderstanding of what a seed does: a seed selects *which* random draw is
   taken, so it only matters when a draw happens at all, and at temperature 0
   none does. The error mattered twice over. It produced advice that would
   have wasted campaign time (set a seed and re-run, expecting different
   attempts), and it shaped `_retry_determinism_note`, whose first version
   returned as soon as a seed was present and so would have *stopped warning*
   about futile retries while they remained futile — a signal that goes quiet
   without the underlying problem being fixed, which is the same failure shape
   as the two mistakes this project has already recorded. Both the code and
   this reasoning were corrected: temperature is checked before the seed and
   reported either way. Recorded rather than deleted, because how the decision
   was reached is part of what this document exists to show.
3. **Task 3's warning must not change any verdict.** It is stored separately
   from `failures` and `passed` is computed from `failures` alone, so a
   warning cannot alter the pass rate earlier results were collected under.

## 7. Added after implementation — one deviation from this plan

`tests/test_run_eval.py` needed a change that was not foreseen above, and it
is worth recording because it is a real finding rather than a tidy-up.

`test_full_batch_runs_four_modes_and_writes_csv` asserted that a four-mode
batch issues eight model calls. That count is attempts-dependent, and
`run_eval` reads `config.QUIZ_MAX_ATTEMPTS`, which is read from `.env`. The
working `.env` on this machine now contains `QUIZ_MAX_ATTEMPTS=3` and
`LLM_TEMPERATURE=0`, so the batch issued sixteen calls and the assertion
failed. The failure was pre-existing and had nothing to do with the prompt 5
changes: it appeared as soon as the environment was configured for the real
evaluation campaign.

The test now pins `QUIZ_MAX_ATTEMPTS` to 1 for the duration of that
assertion, so it tests the batch rather than the developer's environment.

This also confirms, from the working configuration itself, that the campaign
runs with `LLM_TEMPERATURE=0`, no `LLM_SEED` and three attempts, which is
exactly the case where retry cannot recover a format failure and the new
`quiz_retry_note` fires.

**Correction.** This paragraph originally ended by advising that setting
`LLM_SEED` in `.env` would make the Task 1 fix take effect for the campaign.
It would not, for the reason given in the correction to section 6 point 2: at
`LLM_TEMPERATURE=0` decoding is greedy and the seed changes nothing. What the
campaign needs, if retry is to be a real recovery mechanism rather than a
measurement of the same failure three times, is `LLM_TEMPERATURE` above 0;
`LLM_SEED` then keeps that sampled sequence reproducible.

There is a trade-off to decide before changing it, and it belongs in the
Evaluation chapter rather than being settled here. `LLM_TEMPERATURE=0` is what
makes the input-mode comparison reproducible, so raising it to make retry work
reintroduces the run-to-run variance the campaign set out to control. Running
the mode comparison at temperature 0 with `QUIZ_MAX_ATTEMPTS=1`, and any
retry experiment separately above 0, keeps the two claims from interfering.
