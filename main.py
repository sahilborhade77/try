import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import av
import cv2
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
    """Wraps reference signs and provides prediction with DTW distance."""

    def __init__(self, reference_signs: pd.DataFrame) -> None:
        self.reference_signs = reference_signs

    def predict_sign(self, sequence: Dict[str, List[List[float]]]) -> Tuple[str, float]:
        """Return best matching label and DTW distance score (lower is better)."""
        try:
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
        except Exception as exc:
            raise RuntimeError(f"Prediction failed: {exc}") from exc


class SignFrameProcessor(VideoProcessorBase):
    """Processes webcam frames, extracts landmarks, and runs DTW prediction."""

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
    """Load sign references once per session."""
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


def render_homepage() -> None:
    st.subheader("Project Overview")
    st.info(
        "This application recognizes sign gestures in real time by tracking hand landmarks and "
        "matching the live motion sequence with stored references."
    )

    st.markdown("### What is Sign Language Recognition?")
    st.write(
        "Sign Language Recognition converts human gestures into machine-readable labels, "
        "enabling accessible interaction between signers and digital systems."
    )

    st.markdown("### What is MediaPipe?")
    st.write(
        "MediaPipe is a framework for real-time perception. In this project it detects hand landmarks "
        "for every video frame."
    )

    st.markdown("### What is DTW?")
    st.write(
        "Dynamic Time Warping compares time-series sequences even when speed varies. "
        "It is used to find the closest gesture match from the reference library."
    )

    st.divider()
    st.markdown("### Key Features")
    st.success("Real-time webcam detection")
    st.success("Landmark extraction")
    st.success("DTW sequence matching")
    st.success("Instant prediction display")


def render_live_page(recognizer: Optional[DTWRecognizer], init_error: str) -> None:
    st.subheader("Live Recognition")

    if init_error:
        st.warning(init_error)
        return

    if recognizer is None:
        st.warning("Recognizer is not ready.")
        return

    if "result_store" not in st.session_state:
        st.session_state.result_store = LiveResultStore()

    result_store: LiveResultStore = st.session_state.result_store

    if st.session_state.get("recognition_running", False):
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
            st.warning("Webcam stream is not active yet. Check permissions or camera availability.")

        with result_store.lock:
            label = result_store.label
            score = result_store.score
            status = result_store.status
            error = result_store.last_error
            hands_detected = result_store.hands_detected

        col1, col2, col3 = st.columns(3)
        col1.metric("Predicted Sign", label)
        col2.metric("Matching Score (DTW distance)", "-" if score is None else f"{score:.2f}")
        col3.metric("Live Status", status)

        if hands_detected:
            st.success("Hand landmarks detected.")
        else:
            st.info("Show your hand clearly in front of the camera.")

        if error:
            st.warning(error)
    else:
        st.info("Recognition is stopped. Use the sidebar Start button to begin.")


def render_about() -> None:
    st.subheader("About Project")
    st.write(
        "This final-year engineering project demonstrates real-time sign gesture recognition using "
        "MediaPipe landmarks and Dynamic Time Warping for sequence alignment."
    )
    st.write(
        "The system is designed for practical demos and easy deployment on Streamlit Cloud "
        "with webcam-based interaction."
    )


def main() -> None:
    st.set_page_config(page_title="Sign Language Recognition System", layout="wide")

    if "recognition_running" not in st.session_state:
        st.session_state.recognition_running = False

    st.markdown("<h1 style='text-align: center;'>Sign Language Recognition System</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; font-size: 1.1rem;'>DTW-Based Real-Time Gesture Recognition</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    with st.sidebar:
        st.header("Navigation")
        page = st.radio("Go to", ["Home", "Live Recognition", "About Project"])
        st.divider()

        start_clicked = st.button("Start Recognition", use_container_width=True)
        stop_clicked = st.button("Stop Recognition", use_container_width=True)

        if start_clicked:
            st.session_state.recognition_running = True
        if stop_clicked:
            st.session_state.recognition_running = False

        st.divider()
        if st.session_state.recognition_running:
            st.success("Recognition: ON")
        else:
            st.warning("Recognition: OFF")

    with st.spinner("Initializing DTW recognizer..."):
        recognizer, init_error = initialize_recognizer()

    if page == "Home":
        render_homepage()
    elif page == "Live Recognition":
        render_live_page(recognizer, init_error)
    else:
        render_about()

    st.divider()
    st.caption(
        "Developed by Sahil Borhade ,Bhagyshri Sonawane, Anjali Waware,Pranjal Wade | "
        "Final Year Engineering Project"
    )


if __name__ == "__main__":
    main()
