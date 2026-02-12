#!/usr/bin/env python3
"""
Test script to verify MediaPipe imports don't trigger cv2.
"""

try:
    from utils.mediapipe_utils import mediapipe_detection
    print("✓ mediapipe_utils imported successfully")
except ImportError as e:
    print(f"✗ Failed to import mediapipe_utils: {e}")

try:
    import cv2
    print("✗ cv2 is imported (this should not happen)")
except ImportError:
    print("✓ cv2 not imported (good)")

try:
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    print("✓ MediaPipe Tasks API imported successfully")
except ImportError as e:
    print(f"✗ Failed to import MediaPipe Tasks: {e}")

print("Import test completed.")
