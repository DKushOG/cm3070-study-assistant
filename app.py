"""Multimodal AI Study Assistant.

Streamlit interface over the orchestration pipeline. Audio and slide
uploads appear only when the matching module is installed, and every
source can always be pasted manually, so the app runs at every stage of
development. Intermediate outputs stay visible so the user can inspect the
transcript, extracted slide text and combined context before trusting the
generated notes, which is the transparency requirement from the Design
chapter.

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
from modules import audio_stt, slide_vision, slides_ocr, slides_pptx

st.set_page_config(page_title="AI Study Assistant", layout="wide")
st.title("Multimodal AI Study Assistant")
st.caption(
    "Generates structured revision notes and quiz questions from lecture "
    f"materials. Model: {config.OPENAI_MODEL} via {config.OPENAI_BASE_URL}"
)

for key in ("transcript", "slide_text", "notes", "slide_source",
            "slide_text_at_extraction", "transcript_model",
            "transcript_at_extraction"):
    st.session_state.setdefault(key, "")

with st.sidebar:
    st.header("Input sources")
    st.write(
        "Provide any combination of sources. The input mode is derived "
        "from what you supply."
    )
    # Selecting the Whisper size here rather than in .env lets the same audio
    # be run through tiny, base and small without restarting, which is what
    # the word error rate comparison in the Evaluation chapter needs.
    whisper_sizes = audio_stt.WHISPER_MODEL_SIZES
    default_index = (whisper_sizes.index(config.WHISPER_MODEL)
                     if config.WHISPER_MODEL in whisper_sizes else 0)
    whisper_size = st.selectbox(
        "Whisper model size", whisper_sizes, index=default_index,
        help="Defaults to WHISPER_MODEL from .env. Larger sizes are more "
             "accurate and slower.")

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
        # Video containers are accepted because a lecture recording usually
        # arrives as one; Whisper decodes them through ffmpeg, so no
        # conversion step is required of the student.
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
                    # Remember what this size produced, so the label cannot go
                    # stale if the transcript is edited afterwards.
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

NATIVE_PPTX_PATH = "Native PowerPoint text (python-pptx)"
CLASSICAL_PATH = "Classical (pypdf / Tesseract OCR)"
VISION_PATH = f"Vision model ({config.VISION_MODEL})"
SLIDE_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def slide_paths_for(suffix):
    """Return the extraction paths that can run for this file type.

    A pptx reaches the pixel-based paths only through a LibreOffice
    conversion, so those are offered for a deck only when LibreOffice is
    present. The native text path never needs it.
    """
    is_pptx = suffix == slides_pptx.PPTX_SUFFIX
    is_image = suffix in SLIDE_IMAGE_SUFFIXES
    pdf_reachable = suffix == ".pdf" or (
        is_pptx and slides_pptx.libreoffice_available())
    paths = []
    if is_pptx and slides_pptx.is_available():
        paths.append(NATIVE_PPTX_PATH)
    if ((pdf_reachable and (slides_ocr.pdf_available()
                            or slides_ocr.ocr_available()))
            or (is_image and slides_ocr.ocr_available())):
        paths.append(CLASSICAL_PATH)
    if (pdf_reachable or is_image) and slide_vision.is_available():
        paths.append(VISION_PATH)
    return paths


with middle:
    st.subheader("Slides, PowerPoint or PDF")
    # Each available extraction path is offered as a choice, and which paths
    # exist depends on the uploaded file type. The native pptx path keeps
    # slide structure, the classical path reads text only, and the vision path
    # also describes diagrams. Which one produced the current text is shown
    # below for the traceability required by the Design chapter.
    if (slides_ocr.pdf_available() or slides_ocr.ocr_available()
            or slide_vision.is_available() or slides_pptx.is_available()):
        slide_file = st.file_uploader(
            "Slide file", type=["pptx", "pdf", "png", "jpg", "jpeg"])
        chosen_path = None
        if slide_file is not None:
            upload_suffix = Path(slide_file.name).suffix.lower()
            available_paths = slide_paths_for(upload_suffix)
            if not available_paths:
                st.warning(
                    f"No extraction path is available for a '{upload_suffix}' "
                    "file. Paste the slide text instead.")
            elif len(available_paths) > 1:
                chosen_path = st.radio(
                    "Slide extraction method", available_paths,
                    help="Native PowerPoint text keeps slide structure; "
                         "classical OCR reads text only; the vision model "
                         "also describes diagrams, figures and charts.")
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
                    is_pptx = suffix == slides_pptx.PPTX_SUFFIX
                    if chosen_path == NATIVE_PPTX_PATH:
                        # Speaker notes stay off: they are content no other
                        # path can see, so including them would confound the
                        # comparison in the Evaluation chapter.
                        extracted = slides_pptx.extract_pptx_text(
                            tmp_path, include_notes=False)
                    elif chosen_path == VISION_PATH:
                        vision_client = LLMClient(model=config.VISION_MODEL)
                        # Pre-flight before rendering pages and calling the
                        # model once per page, so an unpulled model is
                        # reported up front rather than mid-extraction.
                        reachable, message = vision_client.check_model_available()
                        if not reachable:
                            raise RuntimeError(message)
                        if is_pptx:
                            images = slides_pptx.render_pptx_to_images(
                                tmp_path, dpi=config.VISION_DPI,
                                max_pages=config.VISION_MAX_PAGES)
                            extracted = slide_vision.extract_from_images(
                                images, vision_client)
                        else:
                            extracted = slide_vision.extract_slides(
                                tmp_path, vision_client)
                    elif is_pptx:
                        # Convert once, then reuse the existing PDF path so
                        # nothing downstream needs to know about PowerPoint.
                        with tempfile.TemporaryDirectory() as workspace:
                            pdf_path = slides_pptx.convert_to_pdf(
                                tmp_path, workspace)
                            extracted = slides_ocr.extract_slides(pdf_path)
                    else:
                        extracted = slides_ocr.extract_slides(tmp_path)
                    # Remember the exact text produced, so a later hand edit
                    # can be detected and disclosed rather than silently
                    # credited to the extractor.
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
                    quiz_max_attempts=config.QUIZ_MAX_ATTEMPTS)
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
                # The validation verdict is shown, not hidden: the two format
                # failures documented in the report are the ones being
                # measured, so the user sees when one has occurred.
                validation = result.quiz_validation
                if validation is not None and not validation.passed:
                    st.warning(
                        f"Quiz format check failed after "
                        f"{result.quiz_attempts} attempt(s): "
                        f"{validation.describe()}")
                st.markdown(result.quiz)
        except RuntimeError as error:
            st.error(
                f"{error}\n\nCheck that the model endpoint is running "
                "(for Ollama: 'ollama serve' and 'ollama pull "
                f"{config.OPENAI_MODEL}') and that .env is configured.")
