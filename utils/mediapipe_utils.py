import numpy as np

def mediapipe_detection(image):
    """
    Make hand landmarker prediction on image using MediaPipe Tasks API.

    :param image: Input image (numpy array, RGB)
    :return: Processed image and results
    """
    # Lazy import to avoid cv2 at module level
    import mediapipe.tasks.python.vision as vision
    import mediapipe.tasks.python.core as core

    # Create HandLandmarker
    base_options = core.BaseOptions(model_asset_path='models/hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
    hand_landmarker = vision.HandLandmarker.create_from_options(options)

    # Create vision.Image from numpy array (assuming RGB)
    mp_image = vision.Image(image_format=vision.ImageFormat.SRGB, data=image)

    # Perform detection
    results = hand_landmarker.detect(mp_image)

    # Return original image and results
    return image, results
