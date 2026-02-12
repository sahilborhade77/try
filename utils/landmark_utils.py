import numpy as np

def extract_landmarks(results):
    """
    Extract hand landmarks from MediaPipe results.

    :param results: MediaPipe results object
    :return: Tuple of (pose, left_hand, right_hand) landmarks
    """
    # Pose not used in recognition, set to zeros for compatibility.
    pose = np.zeros(132, dtype=np.float32)

    # 21 landmarks * 3 coords (x, y, z)
    left_hand = np.zeros(63, dtype=np.float32)
    right_hand = np.zeros(63, dtype=np.float32)

    if results is None:
        return pose, left_hand, right_hand

    left_obj = getattr(results, "left_hand_landmarks", None)
    right_obj = getattr(results, "right_hand_landmarks", None)

    if left_obj is not None:
        left_points = getattr(left_obj, "landmark", None)
        if left_points is not None:
            left_hand = np.array([[lm.x, lm.y, lm.z] for lm in left_points], dtype=np.float32).flatten()

    if right_obj is not None:
        right_points = getattr(right_obj, "landmark", None)
        if right_points is not None:
            right_hand = np.array([[lm.x, lm.y, lm.z] for lm in right_points], dtype=np.float32).flatten()

    return pose, left_hand, right_hand
