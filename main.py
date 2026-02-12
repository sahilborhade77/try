import sys

import cv2
import mediapipe as mp

from sign_recorder import SignRecorder
from utils.dataset_utils import load_dataset, load_reference_signs
from utils.mediapipe_utils import mediapipe_detection
from utils.sign_storage import get_available_signs
from utils.voice_output import VoiceOutput
from webcam_manager import WebcamManager


def get_sign_name_input():
    """Get sign name from user input."""
    available_signs = get_available_signs()

    print("\n" + "=" * 60)
    print("NEW SIGN RECORDING")
    print("=" * 60)

    if available_signs:
        print(f"\nExisting signs: {', '.join(available_signs)}")

    try:
        sign_name = input("\nEnter sign name (e.g., 'Hello', 'Thanks', 'Goodbye'): ").strip()
    except EOFError:
        print("No interactive input available for sign name.")
        return None

    if not sign_name:
        print("Sign name cannot be empty.")
        return None

    print(f"Recording new sign: '{sign_name}'")
    return sign_name


def main():
    """Main application loop."""

    print("\n" + "=" * 60)
    print("SIGN LANGUAGE RECOGNITION SYSTEM v2.0")
    print("=" * 60)
    print("\nInitializing system...")

    videos = load_dataset()
    reference_signs = load_reference_signs(videos)

    sign_recorder = SignRecorder(reference_signs, mode="recognize")
    webcam_manager = WebcamManager()
    voice_output = VoiceOutput()

    mode = "recognize"
    current_sign_name = None

    print("\n" + "=" * 60)
    print("KEYBOARD CONTROLS")
    print("=" * 60)
    print("  'r' = Start/Stop Recording")
    print("  'm' = Toggle Mode (RECORD <-> RECOGNIZE)")
    print("  'n' = Record NEW Sign")
    print("  'q' = Quit")
    print("=" * 60)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Cannot open webcam!")
        return

    print("\nWebcam opened")
    print(f"Starting in '{mode.upper()}' mode\n")

    with mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as hands:

        try:
            while cap.isOpened():
                ret, frame = cap.read()

                if not ret or frame is None:
                    print("Failed to read frame from webcam (ret=False or empty frame)")
                    break

                image, results = mediapipe_detection(frame, hands)

                sign_detected, is_recording = sign_recorder.process_results(results)
                sequence_length = len(sign_recorder.recorded_results)

                display_image = webcam_manager.update(
                    frame=image,
                    results=results,
                    sign_detected=sign_detected,
                    is_recording=is_recording,
                    sequence_length=sequence_length,
                    current_mode=mode,
                    current_sign_name=current_sign_name,
                    dtw_distance=sign_recorder.last_dtw_distance,
                )

                cv2.imshow("Sign Language Recognition", display_image)

                pressed_key = cv2.waitKey(1) & 0xFF

                if pressed_key == ord("r"):
                    if is_recording:
                        result = sign_recorder.stop_recording()
                        print("Recording stopped")
                        if result:
                            sign_detected = result
                            print(f"Result: {result}")
                            if mode == "record" and result == "Saved sign":
                                current_sign_name = None
                    else:
                        if mode == "record":
                            if current_sign_name:
                                sign_recorder.record(current_sign_name)
                            else:
                                print("No sign name set. Press 'n' to record a new sign.")
                        else:
                            sign_recorder.record()

                elif pressed_key == ord("m"):
                    if mode == "record":
                        mode = "recognize"
                        current_sign_name = None
                    else:
                        mode = "record"

                    sign_recorder.mode = mode
                    voice_output.reset()
                    print(f"\nSwitched to '{mode.upper()}' mode\n")

                elif pressed_key == ord("n"):
                    if mode != "record":
                        print("Switch to RECORD mode first (press 'm')")
                    else:
                        new_sign_name = get_sign_name_input()
                        if new_sign_name:
                            current_sign_name = new_sign_name
                            sign_recorder.record(current_sign_name)

                elif pressed_key == ord("q"):
                    print("\nClosing application...")
                    sign_recorder.stop_recording()
                    break

                if mode == "recognize" and sign_detected and not is_recording:
                    if sign_detected not in {"Unknown Sign", "No reference signs"}:
                        voice_output.speak_sign(sign_detected)

        except KeyboardInterrupt:
            print("\nInterrupted by user")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            voice_output.cleanup()
            print("Webcam released")
            print("Windows closed")
            print("Voice output cleaned up")
            print("\nProgram closed gracefully\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
