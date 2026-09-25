# Multimodal AI Study Assistant

Turns lecture material into revision notes and quiz questions. Give it a
lecture recording, a slide deck and your own typed notes. It transcribes the
recording, extracts the slide content, combines whatever you supplied into one
labelled context, then generates structured revision notes followed by five
short-answer questions with suggested answers.

Everything runs locally through Ollama. No paid API is required.

Built for CM3070 Final Year Project, University of London, template CM3020
Artificial Intelligence, Project Idea 1.

## What you need

| Software | Required for | Install |
|---|---|---|
| Python 3.10 or newer | Everything | python.org |
| Ollama | The text and vision models | https://ollama.com |
| ffmpeg | Transcribing recordings | Windows: `winget install --id=Gyan.FFmpeg -e` |
| Tesseract | The OCR extraction method only | https://github.com/UB-Mannheim/tesseract/wiki |
| LibreOffice | Reading PowerPoint with the image-based methods | https://www.libreoffice.org |

Only Python and Ollama are needed to start. Every source can be pasted in by
hand, and any module whose dependencies are missing is hidden in the sidebar
rather than breaking the app.

## Setup

**1. Create a virtual environment and install the dependencies.**

Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**2. Install the optional input modules you want.**

```
pip install openai-whisper    # transcribe recordings
pip install pymupdf           # render slide pages for the vision model
pip install python-pptx       # read PowerPoint text directly
pip install pypdf             # read a PDF text layer
pip install pytesseract Pillow  # OCR
```

`pip install -r requirements.txt` installs only the core three. The optional
packages are listed at the bottom of that file as well.

**3. Pull the models.**

```
ollama pull llama3.2
ollama pull qwen2.5vl:7b
```

Then build the text model with a larger context window. Ollama allocates about
4,096 tokens by default and silently truncates a longer prompt, so this step
would matter for longer videos that are uploaded:

```
ollama create llama3.2-ctx16k -f Modelfile
```

**4. Copy the example configuration.**

```
copy .env.example .env
```

On macOS or Linux use `cp .env.example .env`. The defaults point at a local
Ollama on `http://localhost:11434/v1` and need no editing.

**5. Start Ollama and run the app.**

```
ollama serve
streamlit run app.py
```

## Using it

The sidebar takes three inputs, and any combination works.

- **Lecture audio.** Audio or video. The Whisper size is chosen in the sidebar.
- **Slides.** PowerPoint, PDF or an image. Choose an extraction method. Per-slide
  routing is offered first and sends a slide to the vision model only when it
  looks visual, which avoids roughly 40 per cent of model calls.
- **Your notes.** Typed or pasted.

The transcript and the extracted slide text appear on screen and can be edited
before generation. The caption under each says which model produced it and
says so again if you have edited it. Press generate to produce the notes and
the quiz. Every run is saved to `outputs/` with a header recording the model,
the settings, the extraction method and the quiz format check.

## Configuration

All settings are environment variables, read in `config.py` and settable in
`.env`. The ones you are likely to change:

| Setting | Default | Purpose |
|---|---|---|
| `OPENAI_BASE_URL` | `http://localhost:11434/v1` | Endpoint. Any OpenAI compatible server works |
| `OPENAI_API_KEY` | `ollama` | Any value for Ollama |
| `OPENAI_MODEL` | `llama3.2-ctx16k` | Text model |
| `VISION_MODEL` | `qwen2.5vl:7b` | Vision model for slide images |
| `WHISPER_MODEL` | `base` | `tiny`, `base`, `small` or `medium` |
| `OUTPUT_DIR` | `outputs` | Where runs are saved |

`.env.example` lists the rest, including the routing thresholds, the sampling
controls and the context word cap. All of them are optional.

## Running the tests

```
python -m unittest discover -s tests -t .
```

263 tests. They need no model, no network and no optional package. Tests that
depend on an optional package are skipped when it is absent.

## Evaluation scripts

```
python -m evaluation.run_eval --samples samples --out outputs
```

Runs the pipeline over every input mode and writes a CSV with the timings
filled in and the scoring columns blank for marking by hand.

```
python -m evaluation.compare_extractors --input samples/slides.pptx --out outputs
```

Runs all four slide extraction methods over one file and writes their output
and a comparison CSV.

```
python -m evaluation.revalidate_outputs --outputs outputs
```

Re-checks saved runs against the current quiz format validator.

```
python run_ablation.py
```

Measures what per-slide routing saves and what it costs, against an all-vision
extraction.

## Project layout

```
app.py                      Streamlit interface
config.py                   Settings, read from the environment
Modelfile                   Builds llama3.2-ctx16k
run_ablation.py             Routing ablation
core/
  cleaning.py               Text cleaning
  context_builder.py        Labelled context assembly
  generation.py             Prompts, and both quiz paths
  llm_client.py             OpenAI compatible client, text and vision
  orchestrator.py           Pipeline controller, timing, saved output
  provenance.py             Which extractor produced which text
  quiz_validation.py        Quiz format checking
modules/
  audio_stt.py              Whisper transcription
  slide_extraction.py       Chooses and runs an extraction method
  slide_routing.py          Per-slide routing rule
  slide_vision.py           Vision-language slide extraction
  slides_ocr.py             PDF text layer and OCR
  slides_pptx.py            PowerPoint text and page rendering
evaluation/                 Measurement harnesses
tests/                      263 unit and integration tests
docs/
  requirements_traceability.md   Requirements mapped to code and tests
  PIN_CONTEXT.md                 Why the text model is built with num_ctx pinned
```

Development scripts sit at the top level beside these. `probe_context.py`,
`probe_schema.py` and the three `probe_vision_*.py` scripts are the one-off
investigations behind the model and configuration choices. `bakeoff_vision.py`
compares candidate vision models, `wer_check.py` scores Whisper sizes,
`auto_measures.py` computes automatic measures over saved runs,
`run_refresh.py` rebuilds results after a model change and
`verify_prompt5.py` checks a set of changes offline. None of them is imported
by the application.

## Notes

Lecture recordings, slide decks and generated output are not in this
repository. `outputs/` and `.env` are ignored by git.
