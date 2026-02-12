import cv2
from types import SimpleNamespace


def _normalize_hands_results(hands_results):
    """
    Map MediaPipe Hands output to an object exposing:
      - left_hand_landmarks
      - right_hand_landmarks
    """
    left_hand_landmarks = None
    right_hand_landmarks = None

    multi_hand_landmarks = getattr(hands_results, "multi_hand_landmarks", None) or []
    multi_handedness = getattr(hands_results, "multi_handedness", None) or []

    for idx, hand_landmarks in enumerate(multi_hand_landmarks):
        label = None
        if idx < len(multi_handedness):
            classification = getattr(multi_handedness[idx], "classification", None)
            if classification:
                label = getattr(classification[0], "label", None)

        if label == "Left":
            if left_hand_landmarks is None:
                left_hand_landmarks = hand_landmarks
            elif right_hand_landmarks is None:
                right_hand_landmarks = hand_landmarks
        elif label == "Right":
            if right_hand_landmarks is None:
                right_hand_landmarks = hand_landmarks
            elif left_hand_landmarks is None:
                left_hand_landmarks = hand_landmarks
        else:
            if left_hand_landmarks is None:
                left_hand_landmarks = hand_landmarks
            elif right_hand_landmarks is None:
                right_hand_landmarks = hand_landmarks

    return SimpleNamespace(
        left_hand_landmarks=left_hand_landmarks,
        right_hand_landmarks=right_hand_landmarks,
        raw=hands_results,
    )


def mediapipe_detection(image, hands_model):
    """
    Run MediaPipe Hands on a cv2 frame.

    :param image: Input image from cv2 (BGR)
    :param hands_model: MediaPipe Hands model
    :return: Tuple[processed_bgr_image, normalized_hands_results]
    """
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    hands_results = hands_model.process(rgb)
    rgb.flags.writeable = True
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    normalized_results = _normalize_hands_results(hands_results)
    return bgr, normalized_results
