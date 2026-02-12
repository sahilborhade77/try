import numpy as np
import pandas as pd

from models.sign_model import SignModel
from utils.landmark_utils import extract_landmarks
from utils.sign_storage import load_all_sign_sequences, save_sign_sequence


class SignRecorder(object):
    def __init__(
        self,
        reference_signs: pd.DataFrame | None = None,
        seq_len=50,
        mode="recognize",
        dtw_threshold=2000,
    ):
        """
        Initialize SignRecorder.

        :param reference_signs: DataFrame with reference signs (kept for compatibility)
        :param seq_len: Number of frames to record per gesture
        :param mode: "record" to create reference signs, "recognize" to match against saved signs
        :param dtw_threshold: Maximum DTW distance to consider a valid match
        """
        self.is_recording = False
        self.is_saving = False
        self.seq_len = seq_len
        self.mode = mode
        self.dtw_threshold = dtw_threshold

        self.recorded_results = []
        self.current_sign_name = None
        self.last_dtw_distance = None
        self.reference_signs = reference_signs if reference_signs is not None else pd.DataFrame()

        self._refresh_sign_sequences()

        print(f"SignRecorder initialized in '{mode}' mode")
        print(f"DTW threshold set to {dtw_threshold}")
        print(f"Loaded {self.num_loaded_signs} reference signs")

    def _refresh_sign_sequences(self):
        self.sign_sequences = load_all_sign_sequences()
        self.num_loaded_signs = len(self.sign_sequences)

    def record(self, sign_name=None):
        """Start recording in current mode."""
        self.recorded_results = []

        if self.mode == "record":
            if sign_name is None:
                raise ValueError("sign_name is required in record mode")
            self.is_recording = False
            self.current_sign_name = sign_name
            self.is_saving = True
            print(f"Recording '{sign_name}'... Press 'r' again to finish")
            return

        self.is_saving = False
        self.current_sign_name = None
        self.is_recording = True
        print(f"Recording gesture for recognition... (0/{self.seq_len} frames)")

    def process_results(self, results) -> (str, bool):
        """
        Process mediapipe results and manage recording/recognition.

        :return: Tuple of (predicted_text_or_status, is_active_recording)
        """
        if self.is_saving:
            if len(self.recorded_results) < self.seq_len:
                self.recorded_results.append(results)
            else:
                self._save_sign()
                return "Saved sign", self.is_recording or self.is_saving

        if self.is_recording:
            if len(self.recorded_results) < self.seq_len:
                self.recorded_results.append(results)
            else:
                predicted_text = self._compute_distances_and_predict()
                return predicted_text, self.is_recording or self.is_saving

        return "", self.is_recording or self.is_saving

    def save_reference_sign(self, sign_name):
        """Save recorded frames as a reference sign (Streamlit compatibility)."""
        if not self.recorded_results:
            print("No frames to save")
            return

        self.current_sign_name = sign_name
        self.is_saving = True
        self._save_sign()

    def _save_sign(self):
        """Save the recorded gesture sequence to disk."""
        if self.current_sign_name is None:
            print("Error: No sign name set")
            return

        left_hand_list, right_hand_list = [], []
        for results in self.recorded_results:
            _, left_hand, right_hand = extract_landmarks(results)
            left_hand_list.append(left_hand)
            right_hand_list.append(right_hand)

        sign_name = self.current_sign_name
        save_sign_sequence(sign_name, left_hand_list, right_hand_list)
        self._refresh_sign_sequences()

        self.recorded_results = []
        self.is_saving = False
        self.current_sign_name = None

        print(f"Saved sign: {sign_name}")

    def _compute_distances_and_predict(self) -> str:
        """
        Compute DTW distances and predict the best sign.

        :return: Predicted sign name or "Unknown Sign"
        """
        if self.num_loaded_signs == 0:
            print("No reference signs found. Record some signs first using record mode.")
            self.recorded_results = []
            self.is_recording = False
            self.last_dtw_distance = None
            return "No reference signs"

        left_hand_list, right_hand_list = [], []
        for results in self.recorded_results:
            _, left_hand, right_hand = extract_landmarks(results)
            left_hand_list.append(left_hand)
            right_hand_list.append(right_hand)

        recorded_sign = SignModel(left_hand_list, right_hand_list)

        distances = {}
        for sign_name, sequences in self.sign_sequences.items():
            min_distance = float("inf")
            for ref_left, ref_right in sequences:
                ref_sign = SignModel(ref_left.tolist(), ref_right.tolist())
                dist = self._compute_dtw_distance(recorded_sign, ref_sign)
                min_distance = min(min_distance, dist)
            distances[sign_name] = min_distance

        if distances:
            best_sign = min(distances, key=distances.get)
            best_distance = distances[best_sign]
            self.last_dtw_distance = best_distance

            print(f"DTW distances: {distances}")
            print(f"Best match: '{best_sign}' (distance: {best_distance:.2f})")

            if best_distance > self.dtw_threshold:
                print(
                    f"Distance {best_distance:.2f} exceeds threshold {self.dtw_threshold}; classifying as Unknown Sign"
                )
                best_sign = "Unknown Sign"
        else:
            best_sign = "Unknown Sign"
            self.last_dtw_distance = None

        self.recorded_results = []
        self.is_recording = False

        return best_sign

    def _compute_dtw_distance(self, sign1: SignModel, sign2: SignModel) -> float:
        """Compute DTW distance between two sign models."""
        from fastdtw import fastdtw

        seq1_left = sign1.left_hand_list if sign1.has_left_hand else []
        seq1_right = sign1.right_hand_list if sign1.has_right_hand else []
        seq2_left = sign2.left_hand_list if sign2.has_left_hand else []
        seq2_right = sign2.right_hand_list if sign2.has_right_hand else []

        total_distance = 0.0

        def vec_l2(a, b):
            return float(np.linalg.norm(np.array(a) - np.array(b)))

        if sign1.has_left_hand and sign2.has_left_hand and len(seq1_left) > 0 and len(seq2_left) > 0:
            dist, _ = fastdtw(seq1_left, seq2_left, dist=vec_l2)
            total_distance += float(dist)

        if sign1.has_right_hand and sign2.has_right_hand and len(seq1_right) > 0 and len(seq2_right) > 0:
            dist, _ = fastdtw(seq1_right, seq2_right, dist=vec_l2)
            total_distance += float(dist)

        return total_distance if total_distance > 0 else float("inf")

    def stop_recording(self):
        """
        Stop active recording.

        - In record mode: save if frames were captured and sign name is set.
        - In recognize mode: predict if frames were captured.

        :return: Status or prediction string when available.
        """
        if self.is_saving:
            if self.current_sign_name and self.recorded_results:
                self._save_sign()
                return "Saved sign"
            self.is_saving = False
            self.recorded_results = []
            return None

        if self.is_recording:
            if self.recorded_results:
                return self._compute_distances_and_predict()
            self.is_recording = False
            self.recorded_results = []

        return None
