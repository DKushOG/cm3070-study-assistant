# Requirements traceability

Maps each requirement to its status and the specific code or test that
supports the claim. Every status below cites evidence; an uncited claim would
be worse than no claim, since each one has to be defensible in the unseen
written examination.

**How this document was built, and its limitation.** Tables 3.1 and 3.2 live
in the Design chapter, which is not part of this repository. The requirements
below were therefore reconstructed from the codebase, the README and the
design decisions recorded in module docstrings. Rows marked
**needs Ian's input** are ones that could not be recovered from the code, or
where the original wording matters and guessing it would be dishonest. Please
reconcile every row against the real tables before submission, and correct any
requirement whose wording differs from what is written here.

Status values: **Met** / **Partially met** / **Not met** / **Needs Ian's
input**.

Test paths are given as `file::Class::test`. Run the suite with:

    python -m unittest discover -s tests -t . -v

---

## The input-format defect this document exists to record

This is the traceability the Design chapter needs, so it is stated first.

| Requirement | Status before | Status now | Evidence |
|---|---|---|---|
| Chapter 1.3: the system should accept common learning material formats | **Not met** | **Met** | Audio uploader accepted only `mp3`/`wav`/`m4a`, rejecting `mp4`, the dominant lecture recording container; slide uploader accepted only `pdf`/`png`/`jpg`, rejecting `pptx`, the dominant slide format. Now `modules/audio_stt.py::ACCEPTED_AUDIO_FORMATS` covers mp3, wav, m4a, mp4, webm, flac, ogg, pinned by `tests/test_modules_optional.py::AcceptedAudioFormatTests`; `modules/slides_pptx.py` adds native pptx support, covered by `tests/test_slides_pptx.py` |
| Table 3.1, user need "Mixed lecture materials" → "Accept audio, slide or PDF input and optional text notes" | **Partially met** | **Met** | The design response is now implemented for the formats students actually hold. Uploader wiring in `app.py` (audio column and `slide_paths_for`); pptx text extraction `modules/slides_pptx.py::extract_pptx_text` |
| A student's unmodified lecture recording can be uploaded | **Not met** | **Met**, with one caveat | Video containers accepted because Whisper decodes them via ffmpeg (`modules/audio_stt.py` docstring). Caveat: Streamlit's 200 MB default cap would still reject a full-length recording, so `.streamlit/config.toml` raises it to 2000 MB. Needs ffmpeg installed |
| A student's unmodified slide deck can be uploaded | **Met** for text; **Partially met** for pixels | — | Native text extraction needs only `python-pptx` (pure pip). OCR and vision paths for a pptx need LibreOffice; when absent the native path still runs and the interface explains the two ways forward. Evidence: `modules/slides_pptx.py::convert_to_pdf` and `tests/test_slides_pptx.py::LibreOfficeAbsentTests` |

---

## Functional requirements

| # | Requirement (reconstructed) | Status | Evidence |
|---|---|---|---|
| F1 | Transcribe lecture audio to text using a pre-trained speech model | **Met** | `modules/audio_stt.py::transcribe` (Whisper). Availability degrades gracefully: `tests/test_modules_optional.py::AvailabilityTests`. Requires `openai-whisper` + ffmpeg on the machine |
| F2 | Extract text from slides and PDFs | **Met** | `modules/slides_ocr.py::extract_pdf_text` (pypdf text layer) and `::ocr_image` (Tesseract); dispatch tested by `tests/test_modules_optional.py::ExtractSlidesDispatchTests` |
| F3 | Extract slide content using a vision model, including diagrams | **Met** | `modules/slide_vision.py::extract_slides`; the instruction requiring verbatim text *and* diagram description is pinned by `tests/test_slide_vision.py::InstructionTests` |
| F4 | Extract text natively from PowerPoint, preserving slide structure | **Met** | `modules/slides_pptx.py::extract_pptx_text`; slide boundaries, titles and table rows covered by `tests/test_slides_pptx.py::ExtractPptxTextTests` |
| F5 | Accept optional student notes as a third source | **Met** | `app.py` notes column; `core/context_builder.py::build_context` labels it `[Student notes]`; `tests/test_context_builder.py::BuildContextTests` |
| F6 | Combine available sources into one labelled context | **Met** | `core/context_builder.py::build_context_with_report`; ordering and labelling tested in `tests/test_context_builder.py::BuildContextTests` |
| F7 | Derive the input mode from the sources supplied, using the four canonical names | **Met** | `core/context_builder.py::mode_name`; `tests/test_context_builder.py::ModeNameTests::test_four_canonical_modes` |
| F8 | Generate structured revision notes with the six required sections | **Met** | `core/generation.py::REVISION_NOTES_PROMPT` and `::generate_revision_notes`; `tests/test_generation.py::NotesPromptTests` |
| F9 | Generate exactly five open short-answer quiz questions | **Partially met** | Requested by `core/generation.py::QUIZ_PROMPT` (`tests/test_generation.py::QuizPromptTests`), but the model does not always comply. Compliance is now measured rather than assumed: `core/quiz_validation.py::validate_quiz`, `tests/test_quiz_validation.py`. Retry is available but off by default (`QUIZ_MAX_ATTEMPTS`) |
| F10 | Orchestrate the models sequentially, the quiz depending on the generated notes | **Met** | `core/orchestrator.py::run_pipeline`; pinned by `tests/test_orchestrator.py::RunPipelineTests::test_quiz_call_depends_on_generated_notes` (Figure 3.2) |
| F11 | Save every run to a self-documenting file | **Met** | `core/orchestrator.py::save_result` writes model, endpoint, temperature, seed, context cap, slide source, transcript source, quiz validation and truncation; `tests/test_orchestrator.py::SaveResultHeaderTests` and `::SlideSourceTests` |
| F12 | Let the user inspect intermediate output before trusting the result | **Met** | `app.py` shows the transcript, slide text and a "Preview combined context" expander before generation |
| F13 | Allow the model to be swapped without code changes | **Met** | `config.py` (`OPENAI_MODEL`, `OPENAI_BASE_URL`, `VISION_MODEL`); `core/llm_client.py::LLMClient` takes them by injection |
| F14 | Let the user choose the slide extraction path when more than one is available | **Met** | `app.py::slide_paths_for` plus the radio selector; availability varies by uploaded file type |
| F15 | Let the user choose the Whisper model size without editing configuration | **Met** | Sidebar selector in `app.py`, defaulting to `config.WHISPER_MODEL`; sizes from `modules/audio_stt.py::WHISPER_MODEL_SIZES` |
| F16 | Batch-run the four input modes for evaluation | **Met** | `evaluation/run_eval.py::main`; `tests/test_run_eval.py::RunEvalTests` |
| F17 | Compare slide extraction techniques on identical input | **Met** | `evaluation/compare_extractors.py`; pages rendered once and shared (fairness argument in the module docstring); `tests/test_compare_extractors.py` |
| F18 | Report which model produced which text | **Met** | `core/provenance.py`; `tests/test_provenance.py` covers both slide and transcript labels, including the edited case |

## Non-functional requirements

| # | Requirement (reconstructed) | Status | Evidence |
|---|---|---|---|
| N1 | The system runs with no optional dependency installed | **Met** | Every module exposes an availability check and raises `RuntimeError` with an install hint only when actually called: `modules/audio_stt.py`, `slides_ocr.py`, `slide_vision.py`, `slides_pptx.py`. Manual paste always available in `app.py` |
| N2 | A missing heavy dependency produces a clear message, never a crash | **Met** | `modules/slides_pptx.py::convert_to_pdf` names both remedies when LibreOffice is absent; `tests/test_slides_pptx.py::LibreOfficeAbsentTests` asserts the message content |
| N3 | The system runs free and locally, with no paid service required | **Met** | Defaults point at local Ollama (`config.py`); no hosted API integration exists in the codebase |
| N4 | Material stays on the user's machine | **Met** | All processing is local by default; the only network call is to the configured endpoint, which defaults to `localhost:11434` |
| N5 | Results remain comparable across project stages | **Met** | Prompt text unchanged since the Preliminary Project Report (`core/generation.py`); the request built by `LLMClient.chat()` is pinned by `tests/test_llm_client.py::ChatRequestIdentityTests`; every added control is off by default |
| N6 | Runs can be made reproducible | **Met** | `LLM_TEMPERATURE` and `LLM_SEED` are sent only when set: `tests/test_llm_client.py::RequestKwargsTests`. Exposed on the batch runner via `--temperature`/`--seed` |
| N7 | Transparency: the user is told when data is lost or altered | **Met** | Context truncation is reported on screen and recorded in the saved file (`core/context_builder.py::TruncationEvent`, `tests/test_context_builder.py::BuildContextWithReportTests`); provenance labels disclose edits (`core/provenance.py`) |
| N8 | Long inputs cannot silently overrun the model's context window | **Met** | `MAX_CONTEXT_WORDS` (unset by default) with per-source reporting; `tests/test_context_builder.py::BuildContextWithReportTests` |
| N9 | The system is testable without a model, network or optional package | **Met** | 124 tests, 123 passing, 1 skipped, using the injected `tests/fakes.py::FakeLLMClient`. The single skip is a degradation test that can only run when PyMuPDF is absent |
| N10 | Slow or unreachable models are reported before expensive work begins | **Met** | `core/llm_client.py::check_model_available` (lists models, never raises); called before rendering in `evaluation/compare_extractors.py` and before extraction in `app.py`. `tests/test_llm_client.py::PreflightTests`, `tests/test_compare_extractors.py::VisionPreflightTests` |
| N11 | A full-length lecture recording can be uploaded | **Met**, with a documented trade-off | `.streamlit/config.toml` raises the cap to 2000 MB. Trade-off: Streamlit buffers the upload in the server process, so a large file causes a transient memory spike of roughly its own size |
| N12 | Three pre-trained models operating on different data spaces (template requirement) | **Met** | Audio: Whisper (`modules/audio_stt.py`). Text: the language model (`core/generation.py`). Image: the vision-language model (`modules/slide_vision.py`). Note: `pypdf` and `python-pptx` are parsers, not pre-trained models, and do not count toward this requirement |

## Requirements that could not be recovered — needs Ian's input

Each of these is likely to appear in Tables 3.1 or 3.2 but could not be
established from the codebase. Listed rather than guessed.

| # | What is needed | Why it could not be determined |
|---|---|---|
| X1 | The exact wording and numbering of every row in Tables 3.1 and 3.2 | The tables are in the Design chapter, not in this repository. All numbering above (F1…, N1…) is this document's own and will need remapping |
| X2 | Any stated performance target (e.g. maximum acceptable processing time per run) | Timing is measured and saved (`core/orchestrator.py`, `notes_seconds`/`quiz_seconds`), but no threshold exists anywhere in the code to compare against |
| X3 | Any stated accuracy or quality target (e.g. a minimum rubric score) | `evaluation/rubric.py` defines the six criteria, but scoring is manual and no pass mark is encoded |
| X4 | Any target for transcription accuracy (word error rate) | `jiwer` is listed as an optional dependency for this, but no WER threshold or campaign exists in the code yet |
| X5 | Usability or accessibility requirements | Nothing in the codebase states one; the interface is a default-styled Streamlit app |
| X6 | Whether a user study or peer review is a stated requirement, and its acceptance criteria | The README's testing map mentions a peer review with three prepared statements, but the statements themselves are in the report |
| X7 | Requirements explicitly deferred or dropped since the Preliminary Project Report | Cannot be inferred from code; only what exists is visible |

## Known gaps against requirements as reconstructed

Stated plainly, because an examiner will find them anyway.

- **F9** is the one functional requirement not fully met: the model does not
  reliably return exactly five well-formed questions. The system now measures
  the failure rate rather than assuming compliance, and an opt-in retry exists.
  The failure rate itself is not yet reported — it needs a real evaluation run.
- **pptx OCR and vision paths depend on LibreOffice**, which is a system
  install. Without it a deck can still be read natively, so no requirement is
  unmet, but the four-way comparison in section 5.2 needs either LibreOffice or
  a manual PDF export of the deck.
- **ffmpeg and the Tesseract engine are system installs.** The availability
  checks report the Python package only, so `is_available()` can return true
  while a call still fails for a missing binary. The failure is a clear
  `RuntimeError`, not a crash, but it is a real gap between the check and the
  capability.
