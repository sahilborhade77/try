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

    if results.hand_landmarks:
        for i, hand_landmarks in enumerate(results.hand_landmarks):
            # Get handedness for this hand
            handedness = results.handedness[i].category_name if results.handedness else "Right"  # Default to Right if not available

            # Flatten landmarks: [x, y, z] for each of 21 landmarks
            landmarks_flat = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks]).flatten()

            if handedness == "Left":
                left_hand = landmarks_flat
            else:  # Right or unknown
                right_hand = landmarks_flat

    return pose, left_hand, right_hand
