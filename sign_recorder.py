import time
import numpy as np
from collections import Counter

from models.sign_model import SignModel
from utils.landmark_utils import extract_landmarks
from utils.sign_storage import save_sign_sequence, load_all_sign_sequences
from fastdtw import fastdtw  # type: ignore


class SignRecorder:
    """
    Handles:
    - Gesture-based recording (open palm → start, fist → stop)
    - DTW-based recognition
    - Stability-based confirmation
    - Idle-hand handling
    """

    def __init__(
        self,
        reference_signs=None,
        seq_len=40,
        mode="recognize",
        dtw_threshold=2000,
    ):
        # ===== Core state =====
        self.mode = mode
        self.seq_len = seq_len
        self.dtw_threshold = dtw_threshold

        self.is_recording = False
        self.is_saving = False
        self.current_sign_name = None

        self.recorded_results = []
        self.recording_frame_count = 0
        self.max_recording_frames = 45

        # ===== Reference signs =====
        self.sign_sequences = load_all_sign_sequences()
        self.num_loaded_signs = len(self.sign_sequences)

        # ===== Prediction stability =====
        self.prediction_buffer = []
        self.confirmation_frames = 10
        self.last_confirmed_sign = None
        self.last_sign_time = 0
        self.sign_cooldown = 1.5

        # ===== Display =====
        self.last_dtw_distance = None

        print(f"✓ SignRecorder initialized ({mode.upper()})")
        print(f"✓ Loaded {self.num_loaded_signs} reference signs")
        print(f"✓ DTW threshold = {dtw_threshold}")

    # ======================================================================
    # MAIN ENTRY
    # ======================================================================

    def process_results(self, results):
        """
        Called once per frame.
        Returns:
        (predicted_text, is_recording, status_dict)
        """

        status = {
            "hand_visible": False,
            "frame_count": len(self.recorded_results),
            "confidence": 0.0,
        }

        hand_present = self._hand_visible(results)
        status["hand_visible"] = hand_present

        # ===============================
        # RECORD MODE (saving new signs)
        # ===============================
        if self.is_saving:
            self.recorded_results.append(results)
            if len(self.recorded_results) >= self.seq_len:
                self._save_sign()
                return f"Saved: {self.current_sign_name}", False, status
            return "", True, status

        # ===============================
        # START RECORDING (gesture-based)
        # ===============================
        if (
            self.mode == "recognize"
            and hand_present
            and not self.is_recording
            and self._detect_open_palm(results)
        ):
            self._start_recording()
            return "", True, status

        # ===============================
        # RECORDING IN PROGRESS
        # ===============================
        if self.is_recording:
            if not hand_present:
                return self._finalize_recording()

            if self._detect_fist(results):
                return self._finalize_recording()

            self.recorded_results.append(results)
            self.recording_frame_count += 1
            status["frame_count"] = self.recording_frame_count

            if self.recording_frame_count >= self.max_recording_frames:
                return self._finalize_recording()

        return "", self.is_recording, status

    # ======================================================================
    # RECORDING HELPERS
    # ======================================================================

    def record(self, sign_name):
        if self.mode != "record":
            return
        self.current_sign_name = sign_name
        self.is_saving = True
        self.recorded_results = []
        print(f"📹 Recording new sign: {sign_name}")

    def _start_recording(self):
        self.is_recording = True
        self.recorded_results = []
        self.recording_frame_count = 0
        self.prediction_buffer.clear()
        print("🎥 Recording started")

    def _finalize_recording(self):
        if len(self.recorded_results) < 8:
            self._reset()
            return "", False, {}

        predicted = self._recognize()
        self._reset()
        return predicted, False, {}

    def _reset(self):
        self.is_recording = False
        self.recorded_results.clear()
        self.recording_frame_count = 0

    # ======================================================================
    # DTW RECOGNITION
    # ======================================================================

    def _recognize(self):
        if self.num_loaded_signs == 0:
            return "No reference signs"

        # Extract landmarks once
        left_seq, right_seq = [], []
        for r in self.recorded_results:
            _, lh, rh = extract_landmarks(r)
            left_seq.append(lh)
            right_seq.append(rh)

        recorded = SignModel(left_seq, right_seq)

        best_sign = None
        best_dist = float("inf")

        for sign_name, sequences in self.sign_sequences.items():
            for ref_lh, ref_rh in sequences:
                ref = SignModel(ref_lh, ref_rh)
                dist = self._dtw_distance(recorded, ref)
                if dist < best_dist:
                    best_dist = dist
                    best_sign = sign_name

        self.last_dtw_distance = best_dist

        if best_dist > self.dtw_threshold:
            return "Unknown Sign"

        # ===== Stability & cooldown =====
        now = time.time()
        if (
            best_sign == self.last_confirmed_sign
            and now - self.last_sign_time < self.sign_cooldown
        ):
            return ""

        self.last_confirmed_sign = best_sign
        self.last_sign_time = now
        return best_sign

    def _dtw_distance(self, s1: SignModel, s2: SignModel):
        total = 0

        if s1.has_left_hand and s2.has_left_hand:
            d, _ = fastdtw(s1.lh_embedding, s2.lh_embedding)
            total += d

        if s1.has_right_hand and s2.has_right_hand:
            d, _ = fastdtw(s1.rh_embedding, s2.rh_embedding)
            total += d

        return total if total > 0 else float("inf")

    # ======================================================================
    # GESTURE HEURISTICS (FAST & SIMPLE)
    # ======================================================================

    def _hand_visible(self, results):
        hands = getattr(results, "multi_hand_landmarks", None)
        return bool(hands)

    def _detect_open_palm(self, results):
        hands = getattr(results, "multi_hand_landmarks", None)
        if not hands:
            return False

        for hand in hands:
            pts = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark])
            wrist = pts[0]
            tips = pts[[4, 8, 12, 16, 20]]
            avg = np.mean(np.linalg.norm(tips - wrist, axis=1))
            if avg > 0.15:
                return True
        return False

    def _detect_fist(self, results):
        hands = getattr(results, "multi_hand_landmarks", None)
        if not hands:
            return False

        for hand in hands:
            pts = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark])
            wrist = pts[0]
            tips = pts[[4, 8, 12, 16, 20]]
            avg = np.mean(np.linalg.norm(tips - wrist, axis=1))
            if avg < 0.10:
                return True
        return False

    def _finger_distance(self, results, threshold, greater):
        def check(hand):
            if hand is None:
                return False
            pts = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark])
            wrist = pts[0]
            tips = pts[[4, 8, 12, 16, 20]]
            avg = np.mean(np.linalg.norm(tips - wrist, axis=1))
            return avg > threshold if greater else avg < threshold

        return check(results.left_hand_landmarks) or check(results.right_hand_landmarks)

    # ======================================================================
    # SAVE NEW SIGNS
    # ======================================================================

    def _save_sign(self):
        lh_list, rh_list = [], []
        for r in self.recorded_results:
            _, lh, rh = extract_landmarks(r)
            lh_list.append(lh)
            rh_list.append(rh)

        save_sign_sequence(self.current_sign_name, lh_list, rh_list)
        print(f"✓ Saved sign: {self.current_sign_name}")

        self.current_sign_name = None
        self.is_saving = False
        self.recorded_results.clear()
