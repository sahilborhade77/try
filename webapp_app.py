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
            return None, "No saved sign sequences found in data/signs. Record samples first."

        recognizer = DTWRecognizer(reference_signs=reference_signs)
        return recognizer, ""
    except Exception as exc:
        return None, f"Failed to initialize recognizer: {exc}"


def apply_styles() -> None:
    st.markdown(
        """
        <style>
            .main-title {
                font-size: 2.4rem;
                font-weight: 700;
                margin-bottom: 0;
                color: #14213d;
                text-align: center;
            }
            .subtitle {
                text-align: center;
                color: #4a5568;
                margin-top: 0.2rem;
                margin-bottom: 1.2rem;
            }
            .panel {
                background: linear-gradient(120deg, #f8fbff 0%, #edf5ff 100%);
                border: 1px solid #d9e8ff;
                border-radius: 14px;
                padding: 1rem 1.2rem;
                margin-bottom: 1rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_home() -> None:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.subheader("Home")
    st.write(
        "This web app performs real-time sign gesture recognition from webcam video. "
        "It uses MediaPipe to extract hand landmarks and DTW to match live sequences "
        "against your stored sign templates."
    )
    st.write("Use Live Recognition to start webcam-based prediction in your browser.")
    st.markdown("</div>", unsafe_allow_html=True)


def render_live(recognizer: Optional[DTWRecognizer], init_error: str) -> None:
    st.subheader("Live Recognition")

    col_start, col_stop = st.columns(2)
    with col_start:
        if st.button("Start", use_container_width=True):
            st.session_state.recognition_running = True
    with col_stop:
        if st.button("Stop", use_container_width=True):
            st.session_state.recognition_running = False

    if init_error:
        st.warning(init_error)
        return

    if recognizer is None:
        st.warning("Recognizer is not ready.")
        return

    if "result_store" not in st.session_state:
        st.session_state.result_store = LiveResultStore()

    result_store: LiveResultStore = st.session_state.result_store

    if st.session_state.recognition_running:
        st.success("Recognition is running. Allow camera access in your browser.")

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

        c1, c2, c3 = st.columns(3)
        c1.metric("Prediction", label)
        c2.metric("Score", "-" if score is None else f"{score:.2f}")
        c3.metric("Status", status)

        if hands_detected:
            st.success("Hand landmarks detected.")
        else:
            st.info("Show your hand clearly to the camera.")

        if error:
            st.warning(error)
    else:
        st.info("Recognition is stopped.")


def render_about() -> None:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.subheader("About")
    st.write(
        "Sign Language Recognition System is a DTW-based real-time gesture recognizer. "
        "It reuses your existing desktop project logic and wraps it in a browser-based Streamlit interface."
    )
    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title="Sign Language Recognition System", layout="wide")
    apply_styles()

    if "recognition_running" not in st.session_state:
        st.session_state.recognition_running = False

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
    st.divider()

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
