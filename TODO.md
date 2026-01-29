# TODO: Fix MediaPipe ImportError on Streamlit Cloud

## Tasks
- [x] Update utils/mediapipe_utils.py to use mediapipe.tasks.python.vision.HandLandmarker
- [x] Update utils/landmark_utils.py to handle HandLandmarkerResult
- [x] Update app.py to remove mediapipe.solutions import and use HandLandmarker
- [ ] Verify no cv2 import (direct or indirect)
- [ ] Test app loads without libGL error
