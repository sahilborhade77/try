import cv2


def mediapipe_detection(image, holistic):
    """
    Run MediaPipe Holistic on a cv2 frame.

    :param image: Input image from cv2 (BGR)
    :param holistic: MediaPipe holistic model
    :return: Tuple[processed_bgr_image, holistic_results]
    """
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    results = holistic.process(rgb)
    rgb.flags.writeable = True
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr, results
