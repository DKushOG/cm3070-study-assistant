# Requirements traceability

Each requirement, the code that implements it and the tests that cover it.
Test paths are given as `file::Class`. Run the suite with:

    python -m unittest discover -s tests -t .

## Functional

| Requirement | Implemented in | Tested by |
|---|---|---|
| Accept common audio, video, slide and typed-note inputs | `app.py`, `modules/audio_stt.py::ACCEPTED_AUDIO_FORMATS`, `modules/slide_extraction.py::paths_for` | `test_modules_optional.py::AcceptedAudioFormatTests`, `test_slide_extraction.py::OfferedMethodTests` |
| Transcribe recordings and extract slide content using interchangeable methods | `modules/audio_stt.py::transcribe`, `modules/slides_pptx.py::extract_pptx_text`, `modules/slides_ocr.py::extract_slides`, `modules/slide_vision.py::extract_slides`, `modules/slide_extraction.py::extract` | `test_slide_extraction.py::ExtractDispatchTests`, `test_slides_pptx.py`, `test_slide_vision.py`, `test_compare_extractors.py` |
| Route individual slides and report unreadable results | `modules/slide_routing.py::decide` and `::extract_routed`, `modules/slide_vision.py::extract_one_image` | `test_slide_routing.py`, `test_vision_robustness.py` |
| Clean and label sources, then combine the available text | `core/cleaning.py::clean_text`, `core/context_builder.py::build_context_with_report` | `test_cleaning.py`, `test_context_builder.py` |
| Generate structured revision notes and five quiz questions with suggested answers | `core/generation.py`, `core/orchestrator.py::run_pipeline` | `test_generation.py`, `test_orchestrator.py::RunPipelineTests` |
| Validate quiz format and restrict source labels to supplied inputs | `core/quiz_validation.py::validate_quiz`, `core/generation.py::available_basis_choices` and `::build_quiz_schema` | `test_quiz_validation.py`, `test_quiz_structured.py`, `test_revalidate_outputs.py` |
| Show intermediate outputs, final output and processing settings | `app.py`, `core/orchestrator.py::save_result`, `core/provenance.py` | `test_orchestrator.py::SaveResultHeaderTests`, `test_provenance.py` |

## Non-functional

| Requirement | Implemented in | Tested by |
|---|---|---|
| Support independent component testing and configurable local models | `core/llm_client.py::LLMClient` takes its settings by injection, `config.py`, `tests/fakes.py::FakeLLMClient` | `test_llm_client.py`, `test_run_eval.py` |
| Remove temporary uploaded files and retain output metadata for evaluation | `app.py` writes an upload to a temporary file and deletes it after use, `core/orchestrator.py::save_result` | `test_orchestrator.py::SaveResultHeaderTests` |
| Render PowerPoint for image extraction without manual conversion | `modules/slides_pptx.py::convert_to_pdf` | `test_slides_pptx.py::LibreOfficeAbsentTests` |

The last one depends on LibreOffice. Without it the native PowerPoint text
method still runs, the image-based methods are hidden, and the interface says
what is missing.

## Two points worth noting

**Three pre-trained models on three data spaces.** Audio is Whisper, in
`modules/audio_stt.py`. Text is the language model, in `core/generation.py`.
Images are the vision-language model, in `modules/slide_vision.py`. `pypdf` and
`python-pptx` are parsers rather than models and do not count towards this.

**A valid quiz is not necessarily a correct one.** The schema fixes the five
questions, the four fields and the allowed source labels, and the format check
in `core/quiz_validation.py` confirms them. Neither establishes that a
suggested answer is true or that the named source supports it. That was scored
by hand instead.

## Test groups

| Group | Files | Tests |
|---|---|---|
| Context, prompts and provenance | cleaning, context_builder, generation, provenance | 30 |
| Quiz validation and revalidation | quiz_validation, revalidate_outputs | 73 |
| Structured quiz schema | quiz_structured | 14 |
| Model client | llm_client | 13 |
| Input modules and extractor comparison | modules_optional, slides_pptx, slide_vision, compare_extractors | 37 |
| Vision reply handling | vision_robustness | 15 |
| Per-slide routing | slide_routing | 18 |
| Extraction method dispatch | slide_extraction | 17 |
| Unit subtotal | | 217 |
| Integration: full pipeline and batch runner | orchestrator, run_eval | 46 |
| **Total** | | **263** |
