class MockResults:
    """Mock MediaPipe results for Streamlit Cloud compatibility."""
    def __init__(self):
        self.hand_landmarks = []
        self.handedness = []

def mediapipe_detection(image):
    """
    Make hand landmarker prediction on image using MediaPipe Tasks API.
    Returns safe stub to avoid cv2/mediapipe imports.

    :param image: Input image (numpy array, RGB)
    :return: Processed image and results
    """
    # Return original image and empty results to avoid mediapipe imports
    return image, MockResults()
