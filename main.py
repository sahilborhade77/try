import os
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import av
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st
from fastdtw import fastdtw
from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer


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


class SignModel:
    """
    Represents a single sign gesture as a sequence of feature embeddings.
    This keeps your existing DTW feature-extraction logic.
    """

    def __init__(self, left_hand_seq, right_hand_seq):
        self.has_left_hand = self._is_valid(left_hand_seq)
        self.has_right_hand = self._is_valid(right_hand_seq)

        self.lh_embedding = self._build_embedding(left_hand_seq) if self.has_left_hand else []
        self.rh_embedding = self._build_embedding(right_hand_seq) if self.has_right_hand else []

    def _is_valid(self, seq):
        if seq is None:
            return False
        try:
            if len(seq) == 0:
                return False
        except TypeError:
            return False
        return bool(np.any(np.array(seq)))

    def _build_embedding(self, seq):
        embeddings = []
        for frame in seq:
            landmarks = np.array(frame, dtype=np.float32).reshape(21, 3)
            features = self._extract_features(landmarks)
            embeddings.append(features)
        return np.array(embeddings, dtype=np.float32)

    def _extract_features(self, lm):
        wrist = lm[0]
        tips = lm[[4, 8, 12, 16, 20]]
        mcps = lm[[2, 5, 9, 13, 17]]

        tip_distances = np.linalg.norm(tips - wrist, axis=1)

        angles = []
        for tip, mcp in zip(tips, mcps):
            v1 = tip - mcp
            v2 = mcp - wrist
            angle = self._angle_between(v1, v2)
            angles.append(angle)

        palm_spread = np.mean(np.linalg.norm(tips - tips.mean(axis=0), axis=1))

        return np.concatenate([tip_distances, angles, [palm_spread]])

    def _angle_between(self, v1, v2):
        denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-6
        cos_theta = np.dot(v1, v2) / denom
        return np.clip(cos_theta, -1.0, 1.0)


def dtw_distances(recorded_sign: SignModel, reference_signs: pd.DataFrame) -> pd.DataFrame:
    """Compute DTW distances against all reference signs."""
    rec_left_hand = recorded_sign.lh_embedding
    rec_right_hand = recorded_sign.rh_embedding

    distances: List[float] = []
    for _, row in reference_signs.iterrows():
        ref_sign_model = row["sign_model"]

        if (recorded_sign.has_left_hand == ref_sign_model.has_left_hand) and (
            recorded_sign.has_right_hand == ref_sign_model.has_right_hand
        ):
            total_distance = 0.0
            if recorded_sign.has_left_hand:
                total_distance += float(fastdtw(rec_left_hand, ref_sign_model.lh_embedding)[0])
            if recorded_sign.has_right_hand:
                total_distance += float(fastdtw(rec_right_hand, ref_sign_model.rh_embedding)[0])
            distances.append(total_distance)
        else:
            distances.append(float("inf"))

    ranked = reference_signs.copy(deep=True)
    ranked["distance"] = distances
    return ranked.sort_values(by=["distance"])


def landmark_to_array(mp_landmark_list):
    if mp_landmark_list is None:
        return np.zeros((21, 3), dtype=np.float32)

    return np.array([[lm.x, lm.y, lm.z] for lm in mp_landmark_list.landmark], dtype=np.float32)


def normalize_hand_landmarks(landmarks: np.ndarray) -> np.ndarray:
    if landmarks is None or not np.any(landmarks):
        return np.zeros((21, 3), dtype=np.float32)

    points = landmarks.copy()
    wrist = points[0]
    points -= wrist

    palm_size = np.linalg.norm(points[9]) + 1e-6
    points /= palm_size
    return points


def extract_landmarks(results):
    pose = np.zeros(99, dtype=np.float32)
    lh = np.zeros(63, dtype=np.float32)
    rh = np.zeros(63, dtype=np.float32)

    hands = getattr(results, "multi_hand_landmarks", None)
    if hands:
        left_hand = landmark_to_array(hands[0])
        lh = normalize_hand_landmarks(left_hand).reshape(-1)

        if len(hands) > 1:
            right_hand = landmark_to_array(hands[1])
            rh = normalize_hand_landmarks(right_hand).reshape(-1)

    return pose.tolist(), lh.tolist(), rh.tolist()


def mediapipe_detection(image, model):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = model.process(rgb)
    rgb.flags.writeable = True
    output = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return output, results


def draw_landmarks(image, results):
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(232, 254, 255), thickness=1, circle_radius=4),
                mp_drawing.DrawingSpec(color=(255, 249, 161), thickness=2, circle_radius=2),
            )
    return image


class DTWRecognizer:
    """Wrap reference signs and provide DTW prediction."""

    def __init__(self, reference_signs: pd.DataFrame) -> None:
        self.reference_signs = reference_signs

    def predict_sign(self, sequence: Dict[str, List[List[float]]]) -> Tuple[str, float]:
        left_seq = np.array(sequence.get("left_hand", []), dtype=np.float32)
        right_seq = np.array(sequence.get("right_hand", []), dtype=np.float32)

        if left_seq.size == 0 and right_seq.size == 0:
            return "No gesture", float("inf")

        recorded = SignModel(left_seq, right_seq)
        ranked = dtw_distances(recorded, self.reference_signs)
        if ranked.empty:
            return "No reference signs", float("inf")

        best = ranked.iloc[0]
        return str(best["name"]), float(best["distance"])


class FrameInferenceEngine:
    """Stateful per-session frame processor used by video_frame_callback."""

    def __init__(self, recognizer: DTWRecognizer, result_store: LiveResultStore) -> None:
        self.recognizer = recognizer
        self.result_store = result_store
        self.left_buffer: List[List[float]] = []
        self.right_buffer: List[List[float]] = []
        self.lock = threading.Lock()
        self.hands_model = mp.solutions.hands.Hands(
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def process(self, image: np.ndarray) -> np.ndarray:
        with self.lock:
            image, results = mediapipe_detection(image, self.hands_model)
            image = draw_landmarks(image, results)

            _, left_hand, right_hand = extract_landmarks(results)
            hands_present = bool(np.any(left_hand) or np.any(right_hand))

            with self.result_store.lock:
                self.result_store.frames_seen += 1
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
                            self.result_store.last_error = f"Prediction failed: {exc}"
                    else:
                        self.result_store.status = "Buffering gesture sequence"
                else:
                    self.result_store.status = "Waiting for hand landmarks"

                self.result_store.sequence_buffer_len = len(self.left_buffer)

            return image


def get_signs_dir_candidates() -> List[str]:
    return [
        os.path.join("data", "signs"),
        os.path.join("webapp", "data", "signs"),
    ]


def ensure_signs_directory() -> str:
    for path in get_signs_dir_candidates():
        if os.path.exists(path):
            return path

    primary = get_signs_dir_candidates()[0]
    os.makedirs(primary, exist_ok=True)
    return primary


def load_all_sign_sequences() -> Dict[str, List[Tuple[np.ndarray, np.ndarray]]]:
    signs_dir = ensure_signs_directory()
    sign_sequences: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {}

    if not os.path.exists(signs_dir):
        return sign_sequences

    for sign_name in os.listdir(signs_dir):
        sign_path = os.path.join(signs_dir, sign_name)
        if not os.path.isdir(sign_path):
            continue

        sign_sequences[sign_name] = []
        for filename in os.listdir(sign_path):
            if not filename.endswith(".npy"):
                continue

            filepath = os.path.join(sign_path, filename)
            try:
                data = np.load(filepath, allow_pickle=True).item()
                left_hand = data["left_hand"]
                right_hand = data["right_hand"]
                sign_sequences[sign_name].append((left_hand, right_hand))
            except Exception:
                continue

    return sign_sequences


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
            return None, "No trained sign samples found in data/signs. Add saved gestures first."

        return DTWRecognizer(reference_signs=reference_signs), ""
    except Exception as exc:
        return None, f"Failed to initialize recognizer: {exc}"


def reset_live_state() -> None:
    if "result_store" in st.session_state:
        with st.session_state.result_store.lock:
            st.session_state.result_store.status = "Stopped"
    st.session_state.engine = None


def render_home() -> None:
    st.subheader("Project Overview")
    st.info(
        "Real-time sign recognition in the browser using streamlit-webrtc webcam streaming, "
        "MediaPipe hand landmarks, and DTW sequence matching."
    )


def render_live(recognizer: Optional[DTWRecognizer], init_error: str) -> None:
    st.subheader("Live Recognition")

    if init_error:
        st.warning(init_error)
        return

    if recognizer is None:
        st.warning("Recognizer is not ready.")
        return

    if "result_store" not in st.session_state:
        st.session_state.result_store = LiveResultStore()
    if "running" not in st.session_state:
        st.session_state.running = False
    if "engine" not in st.session_state:
        st.session_state.engine = None

    result_store: LiveResultStore = st.session_state.result_store

    col_start, col_stop = st.columns(2)
    with col_start:
        if st.button("Start Recognition", use_container_width=True):
            st.session_state.running = True
    with col_stop:
        if st.button("Stop Recognition", use_container_width=True):
            st.session_state.running = False
            reset_live_state()

    if st.session_state.running:
        if st.session_state.engine is None:
            st.session_state.engine = FrameInferenceEngine(recognizer, result_store)

        engine: FrameInferenceEngine = st.session_state.engine

        def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
            image = frame.to_ndarray(format="bgr24")
            processed = engine.process(image)
            return av.VideoFrame.from_ndarray(processed, format="bgr24")

        ctx = webrtc_streamer(
            key="sign-recognition-stream",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIG,
            media_stream_constraints={"video": True, "audio": False},
            video_frame_callback=video_frame_callback,
            async_processing=True,
            desired_playing_state=True,
        )

        if not ctx.state.playing:
            st.warning("Camera stream is not active yet. Allow browser camera access.")
    else:
        st.info("Recognition is stopped. Click Start Recognition.")

    with result_store.lock:
        label = result_store.label
        score = result_store.score
        status = result_store.status
        error = result_store.last_error
        hands_detected = result_store.hands_detected
        frames_seen = result_store.frames_seen
        buffer_len = result_store.sequence_buffer_len

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted Sign", label)
    c2.metric("DTW Score", "-" if score is None else f"{score:.2f}")
    c3.metric("Buffer Length", buffer_len)
    c4.metric("Frames", frames_seen)

    st.caption(f"Status: {status}")
    if hands_detected:
        st.success("Hand landmarks detected")
    else:
        st.info("Show your hand clearly to the camera")

    if error:
        st.error(error)


def render_about() -> None:
    st.subheader("About")
    st.write(
        "This app runs fully in-browser camera capture on Streamlit Cloud via WebRTC and "
        "uses your DTW gesture matching pipeline for recognition."
    )


def main() -> None:
    st.set_page_config(page_title="Sign Language Recognition System", layout="wide")

    st.title("Sign Language Recognition System")
    st.caption("Browser camera + MediaPipe + DTW")

    if "running" not in st.session_state:
        st.session_state.running = False

    with st.spinner("Loading sign references..."):
        recognizer, init_error = initialize_recognizer()

    with st.sidebar:
        page = st.radio("Navigate", ["Home", "Live Recognition", "About"])

    if page == "Home":
        render_home()
    elif page == "Live Recognition":
        render_live(recognizer, init_error)
    else:
        render_about()


if __name__ == "__main__":
    main()
