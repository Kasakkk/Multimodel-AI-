import csv
import datetime
import io
import json
import os
import random
import tempfile
import time
from typing import Dict, List, Optional, Tuple

import streamlit as st
from transformers import pipeline

from config import DATASET_EVAL_CSV, DATASET_IMAGES_DIR, DEVICE, LOG_FILE_PATH, MODEL_ID
from multimodal_model import answer_question, load_model

FAST_MODEL_ID = "Salesforce/blip-vqa-base"


st.set_page_config(
    page_title="Multi-Modal AI Assistant",
    page_icon="🤖",
    layout="wide",
)


def append_log(entry: Dict[str, str]) -> None:
    logs: List[Dict[str, str]] = []
    if os.path.exists(LOG_FILE_PATH):
        try:
            with open(LOG_FILE_PATH, "r", encoding="utf-8") as f:
                logs = json.load(f)
        except (json.JSONDecodeError, OSError):
            logs = []

    logs.append(entry)
    with open(LOG_FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)


def apply_custom_theme() -> None:
    st.markdown(
        """
        <style>
        .main .block-container {
            padding-top: 1.5rem;
            max-width: 1200px;
        }
        .app-card {
            padding: 1rem 1.2rem;
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-radius: 12px;
            background: rgba(128, 128, 128, 0.06);
            margin-bottom: 1rem;
        }
        .small-muted {
            color: #808080;
            font-size: 0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def ensure_session_state() -> None:
    if "recent_qa" not in st.session_state:
        st.session_state.recent_qa = []
    if "dataset_sample" not in st.session_state:
        st.session_state.dataset_sample = None
    if "eval_history" not in st.session_state:
        st.session_state.eval_history = []


def add_recent_qa(question: str, answer: str, model_used: str, latency_seconds: float) -> None:
    entry = {
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "answer": answer,
        "model_used": model_used,
        "latency_seconds": f"{latency_seconds:.2f}",
    }
    st.session_state.recent_qa.insert(0, entry)
    st.session_state.recent_qa = st.session_state.recent_qa[:10]


def render_recent_history() -> None:
    st.sidebar.header("Recent Q&A")
    history = st.session_state.recent_qa
    if not history:
        st.sidebar.caption("No questions yet.")
        return

    for idx, item in enumerate(history, start=1):
        with st.sidebar.expander(f"{idx}. {item['question'][:40]}"):
            st.write(item["answer"])
            st.caption(
                f"{item['timestamp']} | {item['model_used']} | {item['latency_seconds']}s"
            )


def evaluation_history_csv() -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=[
            "timestamp",
            "image_id",
            "question",
            "ground_truth",
            "model_answer",
            "status",
            "model_used",
        ],
    )
    writer.writeheader()
    for row in st.session_state.eval_history:
        writer.writerow(row)
    return buffer.getvalue()


@st.cache_resource(show_spinner=True)
def get_model() -> Tuple[object, object]:
    return load_model(MODEL_ID, DEVICE)


@st.cache_resource(show_spinner=True)
def get_fast_model() -> Tuple[object, object]:
    return load_model(FAST_MODEL_ID, DEVICE)


@st.cache_resource(show_spinner=False)
def get_asr_pipeline():
    asr_device = 0 if DEVICE == "cuda" else -1
    return pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-tiny",
        device=asr_device,
    )


@st.cache_data(show_spinner=False)
def load_eval_samples() -> List[Dict[str, str]]:
    if not os.path.exists(DATASET_EVAL_CSV):
        return []

    with open(DATASET_EVAL_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def image_path_from_id(image_id: str) -> Optional[str]:
    png_path = os.path.join(DATASET_IMAGES_DIR, f"{image_id}.png")
    jpg_path = os.path.join(DATASET_IMAGES_DIR, f"{image_id}.jpg")
    if os.path.exists(png_path):
        return png_path
    if os.path.exists(jpg_path):
        return jpg_path
    return None


def render_live_assistant() -> None:
    st.subheader("Live Assistant")
    st.caption("Upload an image, ask a question, and get the model's answer locally.")
    st.markdown('<div class="app-card small-muted">Tip: Ask specific questions for better answers. Example: "How many objects are on the table?"</div>', unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1])

    with col_left:
        image_source = st.radio(
            "Image source",
            ["Upload image", "Use camera"],
            horizontal=True,
        )
        uploaded_image = None
        camera_image = None
        if image_source == "Upload image":
            uploaded_image = st.file_uploader(
                "Upload image",
                type=["png", "jpg", "jpeg", "webp"],
                accept_multiple_files=False,
            )
        else:
            st.caption("Allow camera permission in your browser when prompted.")
            camera_image = st.camera_input("Capture live image")

        with st.form("live_assistant_form"):
            response_mode = st.selectbox(
                "Response mode",
                ["Fast", "Balanced", "Detailed"],
                index=0,
                help="Fast is quickest on CPU. Detailed gives richer but slower answers.",
            )
            text_question = st.text_area(
                "Question",
                placeholder="E.g., What is in front of me?",
                height=110,
            )
            outfit_color_mode = st.toggle(
                "Outfit color mode (focus on person clothing details)", value=False
            )
            use_voice = st.toggle("Use voice (optional)", value=False)
            uploaded_audio = None
            if use_voice:
                uploaded_audio = st.file_uploader(
                    "Upload audio question",
                    type=["wav", "mp3", "m4a", "flac", "ogg"],
                    accept_multiple_files=False,
                )

            run_inference = st.form_submit_button(
                "Ask AI", type="primary", use_container_width=True
            )

    with col_right:
        st.markdown("### Model Output")
        output_box = st.container(border=True)

        if run_inference:
            selected_image = camera_image if image_source == "Use camera" else uploaded_image
            if selected_image is None:
                output_box.warning("Please upload an image or capture one from the camera.")
                return

            question = (text_question or "").strip()

            if use_voice and uploaded_audio is not None:
                with st.spinner("Transcribing audio..."):
                    try:
                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=".wav"
                        ) as tmp_audio:
                            tmp_audio.write(uploaded_audio.getbuffer())
                            audio_path = tmp_audio.name

                        transcribed = get_asr_pipeline()(audio_path)
                        voice_text = (transcribed.get("text") or "").strip()
                        if voice_text:
                            question = f"{question} {voice_text}".strip()
                            st.info(f"Recognized voice text: {voice_text}")
                    except Exception as err:
                        output_box.error(
                            f"Voice transcription failed. You can continue with text input. Error: {err}"
                        )
                        return
                    finally:
                        try:
                            os.remove(audio_path)
                        except OSError:
                            pass

            if not question:
                output_box.warning("Please enter a text question or upload voice input.")
                return
            if outfit_color_mode:
                question = (
                    "Describe the visible person and clearly mention clothing/outfit color. "
                    f"User question: {question}"
                )

            tmp_img_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_img:
                    tmp_img.write(selected_image.getbuffer())
                    tmp_img_path = tmp_img.name

                started_at = time.perf_counter()
                with st.spinner("Running local model inference..."):
                    if response_mode == "Fast":
                        active_model, active_processor = get_fast_model()
                    else:
                        active_model, active_processor = get_model()
                    result = answer_question(
                        tmp_img_path,
                        question,
                        active_model,
                        active_processor,
                        DEVICE,
                        response_mode=response_mode.lower(),
                    )
                elapsed = time.perf_counter() - started_at

                answer = result.get("answer", "").strip()
                reasoning = result.get("reasoning", "").strip()
                model_used = result.get("model_used", "unknown")

                if not answer:
                    output_box.error("No answer returned by model.")
                    return

                output_box.success("Inference complete.")
                output_box.markdown(f"**Question:** {question}")
                output_box.markdown(f"**Answer:** {answer}")
                output_box.markdown(f"**Reasoning:** {reasoning}")
                output_box.caption(f"Model: {model_used} | Time: {elapsed:.2f}s")
                add_recent_qa(question, answer, model_used, elapsed)

                append_log(
                    {
                        "timestamp": datetime.datetime.now().isoformat(),
                        "input_type": "streamlit_web",
                        "question": question,
                        "answer": answer,
                        "reasoning": reasoning,
                        "model_used": model_used,
                        "latency_seconds": f"{elapsed:.3f}",
                    }
                )
            except Exception as err:
                output_box.error(f"Inference failed: {err}")
            finally:
                if tmp_img_path:
                    try:
                        os.remove(tmp_img_path)
                    except OSError:
                        pass


def render_dataset_evaluator() -> None:
    st.subheader("Dataset Evaluator")
    st.caption("Pull a random sample from dataset eval CSV and compare model output.")

    if not os.path.exists(DATASET_EVAL_CSV):
        st.error(f"Evaluation CSV not found: {DATASET_EVAL_CSV}")
        return
    if not os.path.exists(DATASET_IMAGES_DIR):
        st.error(f"Evaluation images directory not found: {DATASET_IMAGES_DIR}")
        return

    samples = load_eval_samples()
    if not samples:
        st.warning("No dataset samples found in evaluation CSV.")
        return

    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Pull Random Sample", use_container_width=True):
            st.session_state.dataset_sample = random.choice(samples)

    sample = st.session_state.dataset_sample
    if not sample:
        st.info("Click 'Pull Random Sample' to begin.")
        return

    image_id = (sample.get("image_id") or "").strip()
    question = (sample.get("question") or "").strip()
    ground_truth = (sample.get("answer") or "").strip()
    image_path = image_path_from_id(image_id)

    if not image_path:
        st.error(f"Image not found for image_id={image_id}.")
        return

    c3, c4 = st.columns([1, 1])
    with c3:
        st.image(image_path, caption=f"Image ID: {image_id}", use_container_width=True)
        st.markdown(f"**Question:** {question}")

    with c4:
        if st.button("Evaluate Sample", type="primary", use_container_width=True):
            model, processor = get_model()
            with st.spinner("Evaluating sample..."):
                result = answer_question(image_path, question, model, processor, DEVICE)
            model_answer = (result.get("answer") or "").strip()
            is_exact = model_answer.lower() == ground_truth.lower()
            status = "EXACT MATCH" if is_exact else "MISS"

            st.markdown(f"**Expected Answer:** {ground_truth}")
            st.markdown(f"**Model Answer:** {model_answer}")
            if is_exact:
                st.success(status)
            else:
                st.warning(status)
            model_used = result.get("model_used", "unknown")
            st.caption(f"Model: {model_used}")

            st.session_state.eval_history.append(
                {
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "image_id": image_id,
                    "question": question,
                    "ground_truth": ground_truth,
                    "model_answer": model_answer,
                    "status": status,
                    "model_used": model_used,
                }
            )

    if st.session_state.eval_history:
        st.markdown("### Evaluation History")
        st.dataframe(st.session_state.eval_history, use_container_width=True)
        st.download_button(
            "Export Evaluation Results (CSV)",
            data=evaluation_history_csv(),
            file_name="evaluation_results.csv",
            mime="text/csv",
            use_container_width=True,
        )


def main() -> None:
    ensure_session_state()
    apply_custom_theme()
    render_recent_history()

    st.title("Multi-Modal AI Assistant (Streamlit)")
    st.write(
        "Local visual question answering website with optional voice input and dataset evaluation."
    )

    tab1, tab2 = st.tabs(["Live Assistant", "Dataset Evaluator"])
    with tab1:
        render_live_assistant()
    with tab2:
        render_dataset_evaluator()


if __name__ == "__main__":
    main()
