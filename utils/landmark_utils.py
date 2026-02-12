import numpy as np

def extract_landmarks(results):
    """
    Extract hand landmarks from MediaPipe HandLandmarkerResult.

    :param results: MediaPipe HandLandmarkerResult
    :return: Tuple of (pose, left_hand, right_hand) landmarks
    """
    # Pose not used in recognition, set to zeros
    pose = np.zeros(132)

    left_hand = np.zeros(63)
    right_hand = np.zeros(63)

    if results.left_hand_landmarks:
        left_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks]).flatten()

    if results.right_hand_landmarks:
        right_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks]).flatten()

    return pose, left_hand, right_hand
