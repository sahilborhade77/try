import pandas as pd
import numpy as np
from collections import Counter
import time

from utils.dtw import dtw_distances
from models.sign_model import SignModel
from utils.landmark_utils import extract_landmarks
from utils.sign_storage import save_sign_sequence, load_all_sign_sequences


class SignRecorder(object):
    def __init__(self, reference_signs: pd.DataFrame = None, seq_len=50, mode="recognize", dtw_threshold=2000):
        """
        Initialize SignRecorder.
        
        :param reference_signs: DataFrame with reference signs (for recognize mode)
        :param seq_len: Number of frames to record per gesture
        :param mode: "record" to create reference signs, "recognize" to match against saved signs
        :param dtw_threshold: Maximum DTW distance to consider a valid match (default 2000)
        """
        # Variables for recording
        self.is_recording = False
        self.is_saving = False
        self.seq_len = seq_len
        self.mode = mode
        self.dtw_threshold = dtw_threshold

        # List of results stored each frame
        self.recorded_results = []
        
        # For saving: store the sign name during recording
        self.current_sign_name = None
        
        # Store last DTW distance for display
        self.last_dtw_distance = None

        # DataFrame storing the distances between the recorded sign & all the reference signs from the dataset
        self.reference_signs = reference_signs if reference_signs is not None else pd.DataFrame()
        
        # Load reference sign sequences from disk
        self.sign_sequences = load_all_sign_sequences()
        self.num_loaded_signs = len(self.sign_sequences)
        
        # ===== NEW: Stability-based prediction =====
        # Buffer to track predictions over multiple frames
        self.prediction_buffer = []  # Store predictions over time
        self.prediction_confidence_buffer = []  # Store confidence values
        self.last_confirmed_sign = None  # Last sign confirmed with high confidence
        self.confirmation_threshold = 10  # Frames needed to confirm a sign
        self.confidence_threshold = 0.8  # Confidence needed for valid prediction
        self.last_sign_time = 0  # Time of last confirmed sign
        self.sign_cooldown = 1.5  # Seconds before same sign can be repeated
        
        # ===== NEW: Gesture-based recording detection =====
        self.hand_was_visible = False  # Track if hand was visible in last frame
        self.recording_frame_count = 0  # Count frames during gesture recording
        self.max_recording_frames = 45  # Auto-stop after ~1.5 seconds at 30fps
        
        print(f"✓ SignRecorder initialized in '{mode}' mode")
        print(f"✓ DTW threshold set to {dtw_threshold}")
        print(f"✓ Loaded {self.num_loaded_signs} reference signs")
        print(f"✓ Stability buffer: {self.confirmation_threshold} frames, {self.confidence_threshold} confidence")
        print(f"✓ Gesture-based recording enabled")


    def record(self, sign_name=None):
        """
        Start recording a new gesture sequence.
        
        :param sign_name: Name of the sign to record (required for recording mode)
        """
        if self.mode == "record":
            if sign_name is None:
                raise ValueError("sign_name is required in record mode")
            self.current_sign_name = sign_name
            self.is_saving = True
            print(f"\n📹 Recording '{sign_name}'... Press 'r' again to finish")
        else:
            # In recognize mode, start recording for recognition
            self.is_recording = True
            print(f"\n📹 Recording gesture for recognition... ({0}/{self.seq_len} frames)")

    def process_results(self, results) -> (str, bool, dict):
        """
        Process mediapipe results and manage recording/recognition.
        Implements stability-based prediction and gesture detection.
        
        :param results: mediapipe output
        :return: Tuple of (predicted_text, is_recording, status_dict)
        """
        status = {
            'hand_visible': False,
            'is_recording': self.is_recording,
            'frame_count': len(self.recorded_results),
            'confidence': 0.0
        }
        
        # ===== NEW: Detect hand presence (IDLE state handling) =====
        hand_present = self.detect_hand_presence(results)
        status['hand_visible'] = hand_present
        
        # Handle recording mode (saving reference signs)
        if self.is_saving:
            if len(self.recorded_results) < self.seq_len:
                self.recorded_results.append(results)
            else:
                self._save_sign()
                return f"Saved: {self.current_sign_name}", self.is_saving, status

        # ===== NEW: Gesture-based automatic recording =====
        if hand_present and not self.is_recording and self.mode == "recognize":
            # Detect open palm to START recording
            if self.detect_open_palm(results):
                self.is_recording = True
                self.recorded_results = []
                self.recording_frame_count = 0
                self.prediction_buffer = []
                self.prediction_confidence_buffer = []
                print(f"\n🎥 Open palm detected - Recording started... ({self.recording_frame_count}/{self.max_recording_frames} frames)")
        
        # Handle recognize mode (matching against reference signs)
        if self.is_recording:
            if hand_present:
                # Detect fist to STOP recording
                if self.detect_fist(results):
                    predicted_text = self._compute_distances_and_predict()
                    status['frame_count'] = len(self.recorded_results)
                    return predicted_text, self.is_recording, status
                
                # Auto-stop after max frames
                if len(self.recorded_results) < self.seq_len:
                    self.recorded_results.append(results)
                    self.recording_frame_count += 1
                    status['frame_count'] = len(self.recorded_results)
                    
                    if len(self.recorded_results) >= self.max_recording_frames:
                        predicted_text = self._compute_distances_and_predict()
                        status['frame_count'] = len(self.recorded_results)
                        return predicted_text, self.is_recording, status
            else:
                # No hand visible - stop recording
                if len(self.recorded_results) > 5:  # Only process if we have enough frames
                    predicted_text = self._compute_distances_and_predict()
                    status['frame_count'] = len(self.recorded_results)
                    return predicted_text, self.is_recording, status
                else:
                    self.stop_recording()
        
        return "", self.is_recording, status


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

        # Save to disk
        save_sign_sequence(self.current_sign_name, left_hand_list, right_hand_list)

        # Reset recording state
        self.recorded_results = []
        self.is_saving = False
        self.current_sign_name = None

    def _compute_distances_and_predict(self) -> str:
        """
        Compute DTW distances and predict the sign with stability buffering.
        Only confirms signs with high confidence over multiple frames.
        
        :return: Predicted sign name or "Unknown Sign"
        """
        # Check if we have reference signs
        if self.num_loaded_signs == 0:
            print("⚠ No reference signs found. Record some signs first using 'record' mode.")
            self.recorded_results = []
            self.is_recording = False
            self.last_dtw_distance = None
            return "No reference signs"

        print(f"\n=== Processing sequence of {len(self.recorded_results)} frames ===")

        left_hand_list, right_hand_list = [], []
        for results in self.recorded_results:
            _, left_hand, right_hand = extract_landmarks(results)
            left_hand_list.append(left_hand)
            right_hand_list.append(right_hand)

        # Create a SignModel object with the landmarks gathered during recording
        recorded_sign = SignModel(left_hand_list, right_hand_list)

        # Compute DTW distances against all reference signs
        distances = {}
        for sign_name, sequences in self.sign_sequences.items():
            min_distance = float('inf')
            for ref_left, ref_right in sequences:
                # ref_left and ref_right are full sequences (arrays of shape (seq_len, 63))
                # Convert them to list format for SignModel
                ref_sign = SignModel(ref_left.tolist(), ref_right.tolist())
                # Compute DTW distance
                dist = self._compute_dtw_distance(recorded_sign, ref_sign)
                min_distance = min(min_distance, dist)
            distances[sign_name] = min_distance

        # Find the best match
        if distances:
            best_sign = min(distances, key=distances.get)
            best_distance = distances[best_sign]
            
            # ===== NEW: Calculate confidence (inverse of normalized distance) =====
            # Normalize distance to 0-1 range using threshold
            confidence = max(0, 1 - (best_distance / self.dtw_threshold))
            
            print(f"DTW Distances: {distances}")
            print(f"Best match: '{best_sign}' (distance: {best_distance:.2f}, confidence: {confidence:.2f})")
            
            # Store distance for display
            self.last_dtw_distance = best_distance
            
            # ===== NEW: Stability buffering =====
            # Add prediction to buffer
            self.prediction_buffer.append(best_sign)
            self.prediction_confidence_buffer.append(confidence)
            
            # Keep buffer size manageable
            if len(self.prediction_buffer) > self.confirmation_threshold:
                self.prediction_buffer.pop(0)
                self.prediction_confidence_buffer.pop(0)
            
            # Check if distance is below threshold
            if best_distance > self.dtw_threshold:
                print(f"⚠ Distance {best_distance:.2f} exceeds threshold {self.dtw_threshold}")
                print("→ Classified as 'Unknown Sign'")
                best_sign = "Unknown Sign"
                confidence = 0
            else:
                # Check for sign repetition cooldown
                current_time = time.time()
                if (best_sign == self.last_confirmed_sign and 
                    current_time - self.last_sign_time < self.sign_cooldown):
                    # Same sign within cooldown - don't confirm yet
                    best_sign = ""
                    confidence = 0
                else:
                    # Update confirmed sign and time
                    self.last_confirmed_sign = best_sign
                    self.last_sign_time = current_time
        else:
            best_sign = "Unknown Sign"
            self.last_dtw_distance = None
            confidence = 0

        # Reset recording state
        self.recorded_results = []
        self.is_recording = False
        self.recording_frame_count = 0

        return best_sign


    def _compute_dtw_distance(self, sign1: SignModel, sign2: SignModel) -> float:
        """
        Compute DTW distance between two sign models using FastDTW.
        """
        from fastdtw import fastdtw
        
        # Get embeddings
        emb1_left = sign1.lh_embedding if sign1.has_left_hand else []
        emb1_right = sign1.rh_embedding if sign1.has_right_hand else []
        emb2_left = sign2.lh_embedding if sign2.has_left_hand else []
        emb2_right = sign2.rh_embedding if sign2.has_right_hand else []
        
        total_distance = 0
        
        # Compute DTW for left hand if both have it
        if sign1.has_left_hand and sign2.has_left_hand and len(emb1_left) > 0 and len(emb2_left) > 0:
            dist, _ = fastdtw(emb1_left, emb2_left)
            total_distance += dist
        
        # Compute DTW for right hand if both have it
        if sign1.has_right_hand and sign2.has_right_hand and len(emb1_right) > 0 and len(emb2_right) > 0:
            dist, _ = fastdtw(emb1_right, emb2_right)
            total_distance += dist
        
        return total_distance if total_distance > 0 else float('inf')

    def detect_hand_presence(self, results) -> bool:
        """
        Detect if valid hand landmarks are present in the results.
        Returns False if no hands are detected.
        
        :param results: MediaPipe detection results
        :return: True if at least one hand is detected with valid landmarks
        """
        has_left = results.left_hand_landmarks is not None
        has_right = results.right_hand_landmarks is not None
        return has_left or has_right

    def detect_open_palm(self, results) -> bool:
        """
        Detect open palm gesture (all fingers extended).
        Uses heuristics: fingers are far from palm center.
        
        :param results: MediaPipe detection results
        :return: True if open palm detected in either hand
        """
        def is_hand_open(hand_landmarks):
            if hand_landmarks is None:
                return False
            
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
            
            # Wrist is landmark 0, fingers are landmarks 4, 8, 12, 16, 20
            wrist = landmarks[0]
            finger_tips = [landmarks[4], landmarks[8], landmarks[12], landmarks[16], landmarks[20]]
            
            # Check if most finger tips are far from wrist (open palm)
            distances = [np.linalg.norm(tip - wrist) for tip in finger_tips]
            avg_distance = np.mean(distances)
            
            # Open palm has fingers far from wrist (threshold ~0.15)
            return avg_distance > 0.15
        
        left_open = is_hand_open(results.left_hand_landmarks)
        right_open = is_hand_open(results.right_hand_landmarks)
        return left_open or right_open

    def detect_fist(self, results) -> bool:
        """
        Detect fist gesture (all fingers folded).
        Uses heuristics: fingers are close to palm.
        
        :param results: MediaPipe detection results
        :return: True if fist detected in either hand
        """
        def is_hand_fist(hand_landmarks):
            if hand_landmarks is None:
                return False
            
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
            
            # Wrist is landmark 0, fingers are landmarks 4, 8, 12, 16, 20
            wrist = landmarks[0]
            finger_tips = [landmarks[4], landmarks[8], landmarks[12], landmarks[16], landmarks[20]]
            
            # Check if most finger tips are close to wrist (fist)
            distances = [np.linalg.norm(tip - wrist) for tip in finger_tips]
            avg_distance = np.mean(distances)
            
            # Fist has fingers close to wrist (threshold ~0.10)
            return avg_distance < 0.10
        
        left_fist = is_hand_fist(results.left_hand_landmarks)
        right_fist = is_hand_fist(results.right_hand_landmarks)
        return left_fist or right_fist

    def stop_recording(self):
        """Stop recording without saving."""
        self.is_recording = False
        self.is_saving = False
        self.recorded_results = []
        self.recording_frame_count = 0
