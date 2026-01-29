# TODO: Fix Sign Language Recognition for Streamlit Community Cloud

## Steps to Complete
- [x] Refactor `webcam_manager.py` to use PIL instead of OpenCV for image manipulation and text overlays
- [x] Remove landmark drawing functionality to avoid MediaPipe drawing_utils (which uses cv2)
- [x] Update `requirements.txt` to ensure compatibility (MediaPipe 0.10.5, opencv-python-headless)
- [x] Test app locally for import errors
- [ ] Deploy to Streamlit Cloud and verify no libGL errors
