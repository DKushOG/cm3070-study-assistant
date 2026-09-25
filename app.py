"""Multimodal AI Study Assistant, Streamlit interface.

Uploads appear only when the matching module is installed, and every source can
be pasted by hand instead, so the app runs at any stage of setup. Intermediate
outputs stay on screen so the user can check the transcript, the slide text and
the combined context before trusting the generated notes.

Run with:
    streamlit run app.py
"""
import os
import tempfile
from pathlib import Path

import streamlit as st

import config
from core.context_builder import build_context_with_report, mode_name
from core.llm_client import LLMClient
from core.orchestrator import run_pipeline, save_result
from core.provenance import (describe_slide_provenance,
                             describe_transcript_provenance)
from modules import (audio_stt, slide_extraction, slide_vision, slides_ocr,
                     slides_pptx)

st.set_page_config(page_title="AI Study Assistant", layout="wide")
st.title("Multimodal AI Study Assistant")
st.caption(
    "Generates structured revision notes and quiz questions from lecture "
    f"materials. Text model: {config.OPENAI_MODEL}. Vision model: "
    f"{config.VISION_MODEL}. Endpoint: {config.OPENAI_BASE_URL}"
)

for key in ("transcript", "slide_text", "notes", "slide_source",
            "slide_text_at_extraction", "transcript_model",
            "transcript_at_extraction", "slide_notice"):
    st.session_state.setdefault(key, "")

with st.sidebar:
    st.header("Input sources")
    st.write(
        "Provide any combination of sources. The input mode is derived "
        "from what you supply."
    )
    # Chosen here rather than in .env, so the same audio can be run through
    # tiny, base and small without restarting the app.
    whisper_sizes = audio_stt.WHISPER_MODEL_SIZES
    default_index = (whisper_sizes.index(config.WHISPER_MODEL)
                     if config.WHISPER_MODEL in whisper_sizes else 0)
    whisper_size = st.selectbox(
        "Whisper model size", whisper_sizes, index=default_index,
        help="Defaults to WHISPER_MODEL from .env. Larger sizes are more "
             "accurate and slower.")

    # The structured quiz is the default. The older prompt-only path stays
    # selectable so the two can be compared, and the choice is saved.
    quiz_structured = st.checkbox(
        "Structured quiz (format enforced during decoding)",
        value=config.QUIZ_STRUCTURED_IN_APP,
        help="On: the quiz must match a schema, so it always has five "
             "questions and each names a source you supplied. Off: the "
             "original prompt-only path, checked afterwards.")

    st.subheader("Module status")
    st.write(f"Speech-to-text: {'available' if audio_stt.is_available() else 'not installed'}")
    st.write(f"PDF text extraction: {'available' if slides_ocr.pdf_available() else 'not installed'}")
    st.write(f"Image OCR: {'available' if slides_ocr.ocr_available() else 'not installed'}")
    st.write(f"Slide vision (rendering): {'available' if slide_vision.is_available() else 'not installed'}")
    st.write(f"PowerPoint text: {'available' if slides_pptx.is_available() else 'not installed'}")
    st.write(f"PowerPoint rendering (LibreOffice): {'available' if slides_pptx.libreoffice_available() else 'not found'}")

left, middle, right = st.columns(3)

with left:
    st.subheader("Lecture audio")
    if audio_stt.is_available():
        # Video files are accepted so students can upload lecture recordings
        # directly. Whisper uses ffmpeg to read their audio.
        audio_file = st.file_uploader(
            "Audio or video recording",
            type=audio_stt.ACCEPTED_AUDIO_FORMATS)
        if audio_file is not None and st.button("Transcribe audio"):
            with st.spinner(f"Transcribing with Whisper {whisper_size}..."):
                suffix = Path(audio_file.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(audio_file.getbuffer())
                    tmp_path = tmp.name
                try:
                    transcribed = audio_stt.transcribe(
                        tmp_path, model_size=whisper_size)
                    # Store the Whisper size used for this transcript so it can
                    # be shown correctly later.
                    st.session_state["transcript"] = transcribed
                    st.session_state["transcript_at_extraction"] = transcribed
                    st.session_state["transcript_model"] = whisper_size
                    st.success("Transcript ready. Review it below.")
                except RuntimeError as error:
                    st.error(str(error))
                finally:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
    else:
        st.info("Whisper not installed. Paste a transcript instead.")
    st.session_state["transcript"] = st.text_area(
        "Transcript", value=st.session_state["transcript"], height=220)
    transcript_provenance = describe_transcript_provenance(
        st.session_state["transcript"],
        st.session_state["transcript_model"],
        st.session_state["transcript_at_extraction"])
    if transcript_provenance:
        st.caption(f"Transcript produced by: {transcript_provenance}")

with middle:
    st.subheader("Slides, PowerPoint or PDF")
    # Which methods are offered depends on the uploaded file type. Per-slide
    # routing is listed first and selected by default. The selection logic is
    # kept in modules/slide_extraction.py so it can be tested on its own.
    if (slides_ocr.pdf_available() or slides_ocr.ocr_available()
            or slide_vision.is_available() or slides_pptx.is_available()):
        slide_file = st.file_uploader(
            "Slide file", type=["pptx", "pdf", "png", "jpg", "jpeg"])
        chosen_path = None
        if slide_file is not None:
            upload_suffix = Path(slide_file.name).suffix.lower()
            available_paths = slide_extraction.paths_for(upload_suffix)
            if not available_paths:
                st.warning(
                    f"No extraction path is available for a '{upload_suffix}' "
                    "file. Paste the slide text instead.")
            elif len(available_paths) > 1:
                chosen_path = st.radio(
                    "Slide extraction method", available_paths, index=0,
                    help="Per-slide routing reads each slide from its text "
                         "layer and sends it to the vision model only when "
                         "pictures cover at least a tenth of it. Native "
                         "PowerPoint text keeps slide structure. Classical "
                         "OCR reads text only. The vision model reads every "
                         "slide and also describes diagrams, figures and "
                         "charts.")
            else:
                chosen_path = available_paths[0]
                st.caption(f"Available path for this file: {chosen_path}")
            if (upload_suffix == slides_pptx.PPTX_SUFFIX
                    and not slides_pptx.libreoffice_available()):
                st.info(
                    "LibreOffice was not found, so OCR and vision extraction "
                    "are unavailable for PowerPoint files. Native text "
                    "extraction still works. To use the other paths, install "
                    "LibreOffice or export this deck to PDF and upload that.")
        if slide_file is not None and chosen_path and st.button(
                "Extract slide text"):
            with st.spinner(f"Extracting slide text via {chosen_path}..."):
                suffix = Path(slide_file.name).suffix.lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(slide_file.getbuffer())
                    tmp_path = tmp.name
                try:
                    vision_client = None
                    if slide_extraction.needs_vision_model(chosen_path):
                        vision_client = LLMClient(model=config.VISION_MODEL)
                        # Check the model is installed before any page is
                        # rendered, so a missing model is reported up front.
                        reachable, message = vision_client.check_model_available()
                        if not reachable:
                            raise RuntimeError(message)
                    extracted, notice = slide_extraction.extract(
                        tmp_path, chosen_path, vision_client=vision_client)
                    # Kept in session state so it survives a page rerun.
                    st.session_state["slide_notice"] = notice
                    # Remember the extracted text, so a later edit shows up.
                    st.session_state["slide_text"] = extracted
                    st.session_state["slide_text_at_extraction"] = extracted
                    st.session_state["slide_source"] = chosen_path
                    st.success("Slide text ready. Review it below.")
                except (RuntimeError, ValueError) as error:
                    st.error(str(error))
                finally:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
    else:
        st.info("No slide extractor installed. Paste slide text instead.")
    st.session_state["slide_text"] = st.text_area(
        "Slide text", value=st.session_state["slide_text"], height=220)

slide_provenance = describe_slide_provenance(
    st.session_state["slide_text"],
    st.session_state["slide_source"],
    st.session_state["slide_text_at_extraction"])

with middle:
    if slide_provenance:
        st.caption(f"Slide text produced by: {slide_provenance}")
    if st.session_state["slide_notice"] and st.session_state["slide_text"]:
        st.info(st.session_state["slide_notice"])

with right:
    st.subheader("Student notes")
    st.session_state["notes"] = st.text_area(
        "Notes (optional)", value=st.session_state["notes"], height=305)

transcript = st.session_state["transcript"]
slide_text = st.session_state["slide_text"]
notes = st.session_state["notes"]

current_mode = mode_name(transcript=transcript, slide_text=slide_text, notes=notes)
st.write(f"Current input mode: **{current_mode}**")

if any(value.strip() for value in (transcript, slide_text, notes)):
    preview_context, preview_truncation = build_context_with_report(
        transcript=transcript, slide_text=slide_text, notes=notes,
        max_words=config.MAX_CONTEXT_WORDS)
    if preview_truncation:
        st.warning(
            "Context truncated to fit MAX_CONTEXT_WORDS="
            f"{config.MAX_CONTEXT_WORDS}. "
            + " ".join(event.describe() for event in preview_truncation))
    with st.expander("Preview combined context sent to the model"):
        st.text(preview_context)

if st.button("Generate revision notes and quiz", type="primary"):
    if not any(value.strip() for value in (transcript, slide_text, notes)):
        st.error("Provide at least one source first.")
    else:
        client = LLMClient()
        try:
            with st.spinner("Generating (two sequential model calls)..."):
                result = run_pipeline(
                    client, transcript=transcript, slide_text=slide_text,
                    notes=notes, max_words=config.MAX_CONTEXT_WORDS,
                    slide_source=slide_provenance,
                    transcript_source=transcript_provenance,
                    quiz_max_attempts=config.QUIZ_MAX_ATTEMPTS,
                    quiz_structured=quiz_structured)
            path = save_result(result, output_dir=config.OUTPUT_DIR)
            if result.truncation:
                st.warning(
                    "Context was truncated before generation "
                    f"(MAX_CONTEXT_WORDS={config.MAX_CONTEXT_WORDS}). "
                    + " ".join(event.describe()
                               for event in result.truncation)
                    + " This is recorded in the saved output file.")
            st.success(
                f"Done in {result.total_seconds:.2f}s "
                f"(notes {result.notes_seconds:.2f}s, quiz {result.quiz_seconds:.2f}s). "
                f"Saved to {path}")
            notes_col, quiz_col = st.columns(2)
            with notes_col:
                st.subheader("Revision notes")
                st.markdown(result.notes)
            with quiz_col:
                st.subheader("Quiz questions")
                # Show the validation verdict rather than hide it, because the
                # format failures are what the evaluation measures.
                validation = result.quiz_validation
                quiz_mode = ("structured, format enforced during decoding"
                             if result.quiz_structured
                             else "original prompt-only path")
                st.caption(f"Quiz mode: {quiz_mode}")
                if validation is not None and not validation.passed:
                    st.warning(
                        f"Quiz format check failed after "
                        f"{result.quiz_attempts} attempt(s): "
                        f"{validation.describe()}")
                elif validation is not None:
                    st.success(
                        f"Quiz format check passed: "
                        f"{validation.questions_found} questions, each with "
                        "an answer, a permitted type and a supplied source.")
                st.markdown(result.quiz)
        except RuntimeError as error:
            st.error(
                f"{error}\n\nCheck that the model endpoint is running "
                "(for Ollama: 'ollama serve' and 'ollama pull "
                f"{config.OPENAI_MODEL}') and that .env is configured.")
