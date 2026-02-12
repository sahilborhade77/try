import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import av
import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, WebRtcMode, webrtc_streamer

from models.sign_model import SignModel
from utils.dtw import dtw_distances
from utils.landmark_utils import extract_landmarks
from utils.mediapipe_utils import draw_landmarks, mediapipe_detection
from utils.sign_storage import load_all_sign_sequences


RTC_CONFIG = RTCConfiguration(
    {
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}],
    }
)

MIN_SEQUENCE_LENGTH = 20
MAX_SEQUENCE_LENGTH = 50


@dataclass
class LiveResultStore:
    label: str = "-"
    score: Optional[float] = None
    status: str = "Idle"
    last_error: Optional[str] = None
    frames_seen: int = 0
    hands_detected: bool = False
    sequence_buffer_len: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


class DTWRecognizer:
    """Build predictions from loaded reference signs using existing DTW utility."""

    def __init__(self, reference_signs: pd.DataFrame) -> None:
        self.reference_signs = reference_signs

    def predict_sign(self, sequence: Dict[str, List[List[float]]]) -> Tuple[str, float]:
        left_seq = np.array(sequence.get("left_hand", []), dtype=np.float32)
        right_seq = np.array(sequence.get("right_hand", []), dtype=np.float32)

        if left_seq.size == 0 and right_seq.size == 0:
            return "No gesture", float("inf")

        recorded = SignModel(left_seq, right_seq)
        ranked = dtw_distances(recorded, self.reference_signs.copy(deep=True))

        if ranked.empty:
            return "No reference signs", float("inf")

        best = ranked.iloc[0]
        return str(best["name"]), float(best["distance"])


class SignFrameProcessor(VideoProcessorBase):
    """Process stream frames, extract landmarks, and run DTW prediction."""

    def __init__(self, recognizer: DTWRecognizer, result_store: LiveResultStore) -> None:
        self.recognizer = recognizer
        self.result_store = result_store
        self.left_buffer: List[List[float]] = []
        self.right_buffer: List[List[float]] = []
        self.hands_model = mp.solutions.hands.Hands(
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        image, results = mediapipe_detection(image, self.hands_model)
        image = draw_landmarks(image, results)

        _, left_hand, right_hand = extract_landmarks(results)

        with self.result_store.lock:
            self.result_store.frames_seen += 1
            hands_present = bool(np.any(left_hand) or np.any(right_hand))
            self.result_store.hands_detected = hands_present

            if hands_present:
                self.left_buffer.append(left_hand)
                self.right_buffer.append(right_hand)

                if len(self.left_buffer) > MAX_SEQUENCE_LENGTH:
                    self.left_buffer.pop(0)
                    self.right_buffer.pop(0)

                if len(self.left_buffer) >= MIN_SEQUENCE_LENGTH:
                    sequence = {
                        "left_hand": self.left_buffer,
                        "right_hand": self.right_buffer,
                    }
                    try:
                        label, score = self.recognizer.predict_sign(sequence)
                        self.result_store.label = label
                        self.result_store.score = score
                        self.result_store.status = "Recognizing"
                        self.result_store.last_error = None
                    except Exception as exc:
                        self.result_store.status = "Error"
                        self.result_store.last_error = str(exc)
            else:
                self.result_store.status = "Waiting for hand landmarks"
            self.result_store.sequence_buffer_len = len(self.left_buffer)

        return av.VideoFrame.from_ndarray(image, format="bgr24")


@st.cache_resource(show_spinner=False)
def initialize_recognizer() -> Tuple[Optional[DTWRecognizer], str]:
    try:
        sign_sequences = load_all_sign_sequences()
        rows = {"name": [], "sign_model": [], "distance": []}

        for sign_name, samples in sign_sequences.items():
            for left_hand, right_hand in samples:
                rows["name"].append(sign_name)
                rows["sign_model"].append(SignModel(left_hand, right_hand))
                rows["distance"].append(0.0)

        reference_signs = pd.DataFrame(rows)
        if reference_signs.empty:
            return (
                None,
                "⚠ No trained gesture data found.\n"
                "Please upload or record sign samples before starting live recognition.",
            )

        recognizer = DTWRecognizer(reference_signs=reference_signs)
        return recognizer, ""
    except Exception as exc:
        return None, f"Failed to initialize recognizer: {exc}"


def apply_styles() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background: linear-gradient(180deg, #0e1117 0%, #141a24 100%);
                color: #e5e7eb;
            }
            .main-title {
                font-size: 2.4rem;
                font-weight: 700;
                margin-bottom: 0;
                color: #f3f4f6;
                text-align: center;
            }
            .subtitle {
                text-align: center;
                color: #9ca3af;
                margin-top: 0.2rem;
                margin-bottom: 1.2rem;
            }
            .card {
                background: #111827;
                border: 1px solid #253043;
                border-radius: 12px;
                padding: 1rem 1.1rem;
                min-height: 180px;
            }
            .card h4 {
                margin: 0 0 0.6rem 0;
                color: #f9fafb;
            }
            .card p {
                margin: 0.25rem 0;
                color: #d1d5db;
                line-height: 1.45;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_home() -> None:
    st.subheader("Home")
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        with st.container():
            st.markdown(
                """
                <div class="card">
                    <h4>How It Works</h4>
                    <p>Frames from browser webcam are processed live using MediaPipe hand landmarks.</p>
                    <p>Landmark sequences are compared with stored templates using DTW for prediction.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
    with col2:
        with st.container():
            st.markdown(
                """
                <div class="card">
                    <h4>Using The App</h4>
                    <p>Go to <b>Live Recognition</b> and click <b>Start</b>.</p>
                    <p>Allow browser camera access and perform a trained sign clearly in view.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("---")


def render_live(recognizer: Optional[DTWRecognizer], init_error: str) -> None:
    st.subheader("Live Recognition")

    col_start, col_stop = st.columns(2)
    with col_start:
        if st.button("Start", use_container_width=True):
            st.session_state.running = True
    with col_stop:
        if st.button("Stop", use_container_width=True):
            st.session_state.running = False

    if init_error:
        st.warning(init_error)
        return

    if recognizer is None:
        st.warning("Recognizer is not ready.")
        return

    if "result_store" not in st.session_state:
        st.session_state.result_store = LiveResultStore()

    result_store: LiveResultStore = st.session_state.result_store
    saved_signs_count = (
        int(recognizer.reference_signs["name"].nunique())
        if recognizer is not None and not recognizer.reference_signs.empty
        else 0
    )

    if st.session_state.running:
        st.success("🟢 Recognition Active")
    else:
        st.error("🔴 Recognition Stopped")
    st.markdown("---")

    if st.session_state.running:
        st.caption("Allow camera access in your browser.")

        ctx = webrtc_streamer(
            key="sign-language-recognition",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIG,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
            video_processor_factory=lambda: SignFrameProcessor(recognizer, result_store),
        )

        if not ctx.state.playing:
            st.info("Click Start in the webcam component if the stream is not active.")

        with result_store.lock:
            label = result_store.label
            score = result_store.score
            status = result_store.status
            error = result_store.last_error
            hands_detected = result_store.hands_detected
            buffer_len = result_store.sequence_buffer_len

        c1, c2, c3 = st.columns(3)
        c1.metric("Saved Signs", saved_signs_count)
        c2.metric("Sequence Buffer", buffer_len)
        c3.metric("Current DTW Score", "-" if score is None else f"{score:.2f}")
        st.markdown("---")

        st.markdown(
            """
            <div style="
                text-align:center;
                background:#111827;
                border:1px solid #253043;
                border-radius:12px;
                padding:18px 12px;
                margin:8px 0 14px 0;
            ">
                <div style="font-size:0.95rem;color:#9ca3af;margin-bottom:8px;">Detected Sign</div>
                <div style="font-size:2.2rem;font-weight:800;color:#f9fafb;line-height:1.2;">"""
            + f"""{label}"""
            + """</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if hands_detected:
            st.success("Hand landmarks detected.")
        else:
            st.info("Show your hand clearly to the camera.")

        st.caption(f"Live Status: {status}")
        if error:
            st.warning(error)
    else:
        with result_store.lock:
            stopped_score = result_store.score
            stopped_buffer = result_store.sequence_buffer_len
            stopped_label = result_store.label

        c1, c2, c3 = st.columns(3)
        c1.metric("Saved Signs", saved_signs_count)
        c2.metric("Sequence Buffer", stopped_buffer)
        c3.metric("Current DTW Score", "-" if stopped_score is None else f"{stopped_score:.2f}")
        st.markdown("---")

        st.markdown(
            """
            <div style="
                text-align:center;
                background:#111827;
                border:1px solid #253043;
                border-radius:12px;
                padding:18px 12px;
                margin:8px 0 14px 0;
            ">
                <div style="font-size:0.95rem;color:#9ca3af;margin-bottom:8px;">Detected Sign</div>
                <div style="font-size:2.2rem;font-weight:800;color:#f9fafb;line-height:1.2;">"""
            + f"""{stopped_label}"""
            + """</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_about() -> None:
    st.subheader("About")
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        with st.container():
            st.markdown(
                """
                <div class="card">
                    <h4>Project Summary</h4>
                    <p>Sign Language Recognition System is a real-time browser-based gesture recognition app.</p>
                    <p>It wraps your existing DTW + MediaPipe pipeline for web deployment.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
    with c2:
        with st.container():
            st.markdown(
                """
                <div class="card">
                    <h4>Core Stack</h4>
                    <p>Streamlit + streamlit-webrtc for live browser webcam streaming.</p>
                    <p>MediaPipe for landmarks and DTW matching with your saved sign templates.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown("---")


def main() -> None:
    st.set_page_config(page_title="Sign Language Recognition System", layout="wide")
    apply_styles()

    if "running" not in st.session_state:
        st.session_state.running = False

    st.markdown("<div class='main-title'>Sign Language Recognition System</div>", unsafe_allow_html=True)
    st.markdown(
        "<p class='subtitle'>Real-Time MediaPipe + DTW Gesture Recognition</p>",
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        ["Home", "Live Recognition", "About"],
        horizontal=True,
        label_visibility="visible",
    )
    st.markdown("---")

    with st.spinner("Loading sign references..."):
        recognizer, init_error = initialize_recognizer()

    if page == "Home":
        render_home()
    elif page == "Live Recognition":
        render_live(recognizer, init_error)
    else:
        render_about()


if __name__ == "__main__":
    main()
