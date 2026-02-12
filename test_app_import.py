try:
    import app
    print("✓ app.py imported successfully without mediapipe/cv2 imports")
except ImportError as e:
    print(f"✗ Failed to import app.py: {e}")
