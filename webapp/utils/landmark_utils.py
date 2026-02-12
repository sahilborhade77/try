import cv2
import os
import numpy as np
import pickle as pkl
import mediapipe as mp
from mediapipe.python.solutions import hands as mp_hands

from utils.mediapipe_utils import mediapipe_detection


# ============================================================================
# LANDMARK UTILS (OPTIMIZED)
# ============================================================================
# Improvements:
# 1. Wrist-based normalization (translation invariant)
# 2. Palm-size scaling (scale invariant)
# 3. Fast vectorized operations (no nested loops)
# 4. Safe handling of missing hands
# 5. Ready for ML / LSTM upgrade
# ============================================================================


# -------------------------------
# Basic helpers
# -------------------------------

def landmark_to_array(mp_landmark_list):
    """
    Convert MediaPipe landmark list to NumPy array (21, 3).
    """
    if mp_landmark_list is None:
        return np.zeros((21, 3), dtype=np.float32)

    return np.array(
        [[lm.x, lm.y, lm.z] for lm in mp_landmark_list.landmark],
        dtype=np.float32
    )


# -------------------------------
# Normalization (CORE LOGIC)
# -------------------------------

def normalize_hand_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """
    Normalize hand landmarks to be user- and distance-invariant.

    Steps:
    1. Use wrist (landmark 0) as origin
    2. Scale by palm size (distance wrist ↔ middle MCP)
    3. Return normalized (21, 3) landmarks

    :param landmarks: np.ndarray of shape (21, 3)
    :return: normalized np.ndarray of shape (21, 3)
    """

    # No hand detected
    if landmarks is None or not np.any(landmarks):
        return np.zeros((21, 3), dtype=np.float32)

    landmarks = landmarks.copy()

    # ---- Translation invariance ----
    wrist = landmarks[0]
    landmarks -= wrist

    # ---- Scale invariance ----
    # Palm size = distance wrist ↔ middle finger MCP (landmark 9)
    palm_size = np.linalg.norm(landmarks[9]) + 1e-6
    landmarks /= palm_size

    return landmarks


# -------------------------------
# Landmark extraction (MAIN API)
# -------------------------------

def extract_landmarks(results):
    """
    Extract and normalize pose, left-hand, and right-hand landmarks.

    Returns:
    - pose      : (99,)  flattened pose landmarks (or zeros)
    - left_hand : (63,)  normalized left-hand landmarks
    - right_hand: (63,)  normalized right-hand landmarks
    """

    # ---- Pose (kept for compatibility, not used heavily) ----
    pose = np.zeros(99, dtype=np.float32)

    # ---- Hands (MediaPipe Hands uses multi_hand_landmarks) ----
    lh = np.zeros(63, dtype=np.float32)
    rh = np.zeros(63, dtype=np.float32)

    hands = getattr(results, "multi_hand_landmarks", None)
    if hands:
        left_hand = landmark_to_array(hands[0])
        left_hand = normalize_hand_landmarks(left_hand).reshape(-1)
        lh = left_hand

        if len(hands) > 1:
            right_hand = landmark_to_array(hands[1])
            right_hand = normalize_hand_landmarks(right_hand).reshape(-1)
            rh = right_hand

    return pose.tolist(), lh.tolist(), rh.tolist()


# ============================================================================
# DATASET UTILITIES (OFFLINE USE)
# ============================================================================

def save_landmarks_from_video(video_name):
    """
    Extract and save landmarks from a video file.
    Used only for dataset creation (NOT real-time).
    """

    landmark_list = {"pose": [], "left_hand": [], "right_hand": []}
    sign_name = video_name.split("-")[0]

    cap = cv2.VideoCapture(
        os.path.join("data", "videos", sign_name, video_name + ".mp4")
    )

    with mp_hands.Hands(
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image, results = mediapipe_detection(frame, hands)
            pose, left_hand, right_hand = extract_landmarks(results)

            landmark_list["pose"].append(pose)
            landmark_list["left_hand"].append(left_hand)
            landmark_list["right_hand"].append(right_hand)

        cap.release()

    # ---- Directory setup ----
    base_path = os.path.join("data", "dataset", sign_name, video_name)
    os.makedirs(base_path, exist_ok=True)

    save_array(landmark_list["pose"], os.path.join(base_path, f"pose_{video_name}.pickle"))
    save_array(landmark_list["left_hand"], os.path.join(base_path, f"lh_{video_name}.pickle"))
    save_array(landmark_list["right_hand"], os.path.join(base_path, f"rh_{video_name}.pickle"))


# -------------------------------
# Pickle helpers
# -------------------------------

def save_array(arr, path):
    with open(path, "wb") as f:
        pkl.dump(arr, f)


def load_array(path):
    with open(path, "rb") as f:
        return np.array(pkl.load(f))
