# TODO: Fix Streamlit Cloud Bug - Lazy MediaPipe Imports

- [ ] Update utils/mediapipe_utils.py: Remove top-level MediaPipe imports and add lazy import in mediapipe_detection function
- [ ] Update app.py: Add load_hand_landmarker function with lazy import and model creation
- [ ] Run test_imports.py to confirm cv2 is not imported
- [ ] Verify app starts without triggering cv2
