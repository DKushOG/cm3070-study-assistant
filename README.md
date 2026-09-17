# Multimodal AI Study Assistant (CM3070 Final Project)

Generates structured revision notes and quiz questions from lecture
materials by orchestrating several AI components: speech-to-text for
lecture audio, text extraction for slides, and a language model for
generation. Built for the University of London CM3070 Final Year Project,
template CM3020 Artificial Intelligence, Project Idea 1, Orchestrating AI
models to achieve a goal.

The generation pipeline preserves the exact prompts and behaviour of the
prototype evaluated in the Preliminary Project Report, so results remain
comparable across project stages. The speech-to-text and slide extraction
modules are the planned Phase A and Phase B extensions and degrade
gracefully when their dependencies are not installed, which keeps the app
runnable and demonstrable at every stage of development.

## Project structure

    app.py                     Streamlit interface
    config.py                  Environment-driven settings (.env supported)
    core/
      cleaning.py              Text cleaning (pure functions)
      context_builder.py       Labelled context assembly and mode naming
      generation.py            Prompts and the two sequential model calls
      llm_client.py            OpenAI compatible endpoint client (text+vision)
      orchestrator.py          Pipeline controller, timing, output saving
      provenance.py            Which extractor produced the slide text
      quiz_validation.py       Quiz format checking (pure functions)
    modules/
      audio_stt.py             Whisper transcription (Phase A)
      slides_ocr.py            PDF text extraction and image OCR (Phase B)
      slide_vision.py          Vision-language slide extraction (third model)
      slides_pptx.py           Native PowerPoint text, optional rendering
    docs/
      requirements_traceability.md  Requirement to evidence mapping
    evaluation/
      rubric.py                The six scoring criteria
      run_eval.py              Batch runner for the four-mode comparison
      compare_extractors.py    Slide extractor comparison harness (5.2)
      revalidate_outputs.py    Re-check saved runs against the validator
    tests/                     Unit and integration tests (no model needed)
    samples/                   Bayes theorem sample materials from the PPR
    outputs/                   Generated outputs and scoring sheets

## Quick start

1. Create and activate a virtual environment.

   Windows PowerShell:

       py -m venv .venv
       .venv\Scripts\Activate.ps1

   macOS or Linux:

       python3 -m venv .venv
       source .venv/bin/activate

2. Install the core dependencies:

       pip install -r requirements.txt

3. Start a free local model with Ollama:

       ollama pull llama3.2
       ollama serve

   The defaults already point at Ollama. Copy .env.example to .env to
   customise. To use a hosted OpenAI compatible endpoint instead, change
   OPENAI_BASE_URL, OPENAI_API_KEY and OPENAI_MODEL. No code changes are
   needed to swap models, which is what makes the stronger-model
   comparison in the Evaluation chapter a configuration change rather
   than a rewrite.

4. Run the app:

       streamlit run app.py

Steps 1 to 4 are the whole minimum setup. Nothing below this point is
needed to start the app: with no optional package installed, every source
can still be pasted in by hand and the pipeline runs normally. Add the
optional modules only for the features you want to try.

## Enabling the input modules

Each module checks its own dependencies and degrades gracefully, so an
uninstalled module disables its upload box and leaves manual paste
working rather than breaking the app. The sidebar shows what is live.

Phase A, speech-to-text:

    pip install openai-whisper

Also install ffmpeg (Windows: winget install --id=Gyan.FFmpeg -e, then
restart the terminal). The first transcription downloads the Whisper
model weights, so allow time and disk space on first use.

Phase B, slides and PDFs:

    pip install pypdf pytesseract Pillow

For image OCR also install the Tesseract engine itself (on Windows use
the UB Mannheim installer, https://github.com/UB-Mannheim/tesseract/wiki,
and make sure tesseract.exe is on PATH). Digital PDFs need only pypdf.

Phase B alternative, vision-based slide extraction (the project's third
pre-trained model, operating on image data):

    pip install pymupdf
    ollama pull qwen3-vl:4b

PyMuPDF renders each PDF page to an image and the vision model reads it,
transcribing the slide text and describing diagrams, figures and charts.
That last part is the capability classical OCR lacks and is what the
extractor comparison in the Evaluation chapter measures. PyMuPDF is pure
pip with no system dependency, unlike poppler-based alternatives.

When more than one slide extractor is available the app lets you choose
between them and states underneath the slide text which path produced it.
If you then edit that text by hand, the caption says so rather than
crediting the extractor for your wording.

PowerPoint decks, native text extraction:

    pip install python-pptx

Reads the deck's own structure, so slide boundaries, titles and table
rows survive where OCR would flatten a rendered page. python-pptx is a
parser, in the same category as pypdf: it reads the file format and does
no inference, so it is not one of the project's three pre-trained models.
Speaker notes are extracted only on request and are off by default,
because no other extraction path can see them and including them would
make this path look better simply by having more material.

To run the OCR or vision paths on a pptx, the deck must first become
pixels, and nothing in pure Python renders PowerPoint faithfully. If
LibreOffice is installed it is detected automatically (on PATH or at
C:\Program Files\LibreOffice\program\soffice.exe) and used headlessly to
convert the deck to PDF. **Without LibreOffice the native text path still
works**; the app says so and suggests exporting the deck to PDF from
PowerPoint instead. LibreOffice is never required.

Accepted upload formats:

    Recording   mp3, wav, m4a, mp4, webm, flac, ogg
    Slides      pptx, pdf, png, jpg, jpeg

Video containers are accepted because a lecture recording usually arrives
as video; Whisper decodes them through ffmpeg, so no conversion step is
needed. Streamlit's default 200 MB upload cap is raised to 2000 MB in
.streamlit/config.toml, since a 50 minute recording is commonly 375 to
1100 MB. The trade-off is that Streamlit buffers an upload in the server
process, so a large file causes a transient memory spike of about its own
size.

Availability summary:

    Feature                  Needs (pip)          Needs (system)
    Manual paste             nothing              nothing
    Transcription            openai-whisper       ffmpeg
    PDF text layer           pypdf                nothing
    Image OCR                pytesseract, Pillow  Tesseract engine
    Vision slide reading     pymupdf              an Ollama vision model
    PowerPoint text          python-pptx          nothing
    PowerPoint to pixels     pymupdf              LibreOffice

## Configuration

Every setting is an environment variable, read in config.py and settable
in .env (copy .env.example). All the optional controls are unset or
neutral by default, so the default configuration reproduces the exact
behaviour evaluated in the Preliminary Project Report.

    OPENAI_BASE_URL      Endpoint (default local Ollama)
    OPENAI_API_KEY       API key (any value for Ollama)
    OPENAI_MODEL         Text generation model (default llama3.2)
    WHISPER_MODEL        Whisper size: tiny, base, small, medium
    OUTPUT_DIR           Where runs are saved (default outputs)

    LLM_TEMPERATURE      Unset by default. When unset it is not sent to
                         the model at all. Set (e.g. 0) with LLM_SEED to
                         make runs repeatable when separating an input
                         mode effect from sampling noise.
    LLM_SEED             Unset by default. Sampling seed, as above.
    LLM_TIMEOUT          Seconds per model call, default 600 (the OpenAI
                         client's own default, so unchanged behaviour).

    MAX_CONTEXT_WORDS    Unset by default, meaning no limit. When set,
                         each source is independently capped to this many
                         words; a real 50 minute transcript is around
                         7,000 words and can overrun a local model's
                         context. Truncation is always reported on screen
                         and recorded in the saved output, never silent.

    VISION_MODEL         Vision model for slide reading (qwen3-vl:4b)
    VISION_DPI           Page render resolution (default 150)
    VISION_MAX_PAGES     Page cap per deck (default 20), so a long deck
                         cannot run away with one model call per page

    QUIZ_MAX_ATTEMPTS    Default 1, meaning generate once and never
                         retry: today's exact behaviour. Higher values
                         regenerate a quiz that fails format validation,
                         using the identical unchanged prompt. Validation
                         itself always runs, so the failure rate is
                         measured even on the default setting.

## Running the tests

    python -m unittest discover -s tests -t . -v

The suite runs without any model, network access or optional packages.
It covers the cleaning functions, context building, prompt construction,
the reproducibility controls, context truncation reporting, slide text
provenance, quiz format validation and retry, the vision request format,
the extractor comparison harness, and the full pipeline using an injected
fake client, including the check that the quiz call receives the
generated notes (the sequential orchestration in Figure 3.2 of the
report). Whisper and OCR smoke tests
run only when those packages are installed, so run the suite again on
the development machine after enabling Phase A and Phase B.

## Running the evaluation batch

    python -m evaluation.run_eval --samples samples --out outputs

Runs all four input modes over the sample files, saves each output, and
writes outputs/evaluation_scores.csv with timing filled in and rubric
columns left blank for human scoring. Point --samples at other topic
folders to extend the evaluation across modules, which the Evaluation
chapter needs.

Options for a controlled comparison:

    --repeats 3              Run each mode three times, one CSV row per
                             run, so a mode can be reported as a mean
                             instead of a single sample
    --temperature 0 --seed 42  Pass the reproducibility controls through
    --slide-source "Tesseract OCR"  Label which extractor produced the
                             slide text sample, so runs fed by different
                             extractors can be told apart later

The CSV records model, temperature, seed and slide source alongside the
timings, so a results file is self-describing months later. It also
records the quiz format measurement (questions_found, returned_five,
attempts, validation_passed), which replaces the "Returned five
questions" column previously filled in by hand.

## Revalidating runs saved earlier

    python -m evaluation.revalidate_outputs --outputs outputs

Re-reads every saved run, extracts the quiz section and applies the
current validator, so runs generated before a validator correction can be
restated without spending model time regenerating them. Prints a verdict
per file plus the proportion of runs passing, and writes
outputs/revalidated_quizzes.csv with the fault kinds counted separately
(labels absent, labels present but blank, prompt placeholders echoed
back). Files with no quiz section are skipped with a message rather than
stopping the batch.

## Comparing slide extractors

    python -m evaluation.compare_extractors --input samples/slides.pptx --out outputs

Runs every available extraction path over the same file: native pptx
text, the pypdf text layer, Tesseract OCR, and the vision model. A pptx
is converted to PDF once with LibreOffice, after which the other three
paths run exactly as they do for a PDF, giving four comparable techniques
on one deck. Without LibreOffice only the native path runs, and the rest
are skipped with a message. Each PDF page is rendered to
an image once with PyMuPDF and those identical images are shared between
the Tesseract and vision paths, so all image-based paths see exactly the
same pixels and any quality difference is the model's rather than the
input's. Unavailable paths are skipped with a warning, and the vision
model is pre-flighted first so an unpulled model is reported up front,
naming the pull command, instead of failing after the slow work.

Each path's text is saved to its own file for inspection and
outputs/extractor_comparison.csv is written with the score columns left
blank for manual marking, matching the OCR 5.2 sheet of the evaluation
workbook, plus automatic character count, word count, timing, model and
output file columns.

## Development roadmap

    Phase A  Wire Whisper transcription into the app        Gantt task 13
    Phase B  PDF and OCR slide extraction                   Gantt task 13
    Phase C  Full pipeline integration and interface polish Gantt task 14
    Phase D  Component and end-to-end evaluation campaign   Gantt task 15
    Phase E  Stronger-model comparison and refinement       Gantt task 15
    Phase F  Final report, packaging, presentation video    Gantt tasks 16 to 18

## Testing map (verification and validation)

Unit testing: the pure-function and prompt tests in tests/. Integration
testing: the orchestrator tests with the fake client, plus smoke runs
against the live Ollama endpoint on the development machine. System
testing: a manual walkthrough of the Streamlit app against the
functional requirements table in the Design chapter. Acceptance testing:
the testing peer review, using the three agree or disagree statements
prepared in the report scaffold, plus rubric scoring by the developer
acting as the target user.
