import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def mediapipe_detection(image, model):
    """
    Make hand landmarker prediction on image using MediaPipe Tasks API.

    :param image: Input image (numpy array, RGB)
    :param model: MediaPipe HandLandmarker model
    :return: Processed image and results
    """
    # Create vision.Image from numpy array (assuming RGB)
    mp_image = vision.Image(image_format=vision.ImageFormat.SRGB, data=image)

    # Perform detection
    results = model.detect(mp_image)

    # Return original image and results
    return image, results
