import cv2
import mediapipe as mp
from mediapipe.python.solutions import hands as mp_hands
import sys
import time

from utils.dataset_utils import load_dataset, load_reference_signs
from utils.mediapipe_utils import mediapipe_detection
from utils.sign_storage import get_available_signs
from utils.voice_output import VoiceOutput
from sign_recorder import SignRecorder
from webcam_manager import WebcamManager


# ============================================================================
# SIGN LANGUAGE RECOGNITION SYSTEM v3.1 (OPTIMIZED)
# ============================================================================
# Optimizations:
# 1. MediaPipe Hands only (no face/pose overhead)
# 2. FPS limiting for smooth performance
# 3. Idle-frame short-circuit (skip heavy logic)
# 4. Non-blocking voice output (cooldown)
# 5. Stable real-time execution on CPU
# ============================================================================


def get_sign_name_input():
    """Get sign name from user input."""
    available_signs = get_available_signs()

    print("\n" + "=" * 60)
    print("📝 NEW SIGN RECORDING")
    print("=" * 60)

    if available_signs:
        print(f"\nExisting signs: {', '.join(available_signs)}")

    sign_name = input("\nEnter sign name (e.g., 'Hello', 'Thanks'): ").strip()

    if not sign_name:
        print("⚠ Sign name cannot be empty.")
        return None

    print(f"✓ Recording new sign: '{sign_name}'")
    return sign_name


def main():
    print("\n" + "=" * 60)
    print("🤟 SIGN LANGUAGE RECOGNITION SYSTEM v3.1")
    print("=" * 60)
    print("\nInitializing system...")

    # ========================
    # Load dataset & references
    # ========================
    videos = load_dataset()
    reference_signs = load_reference_signs(videos)

    # ========================
    # Initialize components
    # ========================
    sign_recorder = SignRecorder(reference_signs, mode="recognize")
    webcam_manager = WebcamManager()
    voice_output = VoiceOutput()

    mode = "recognize"
    current_sign_name = None

    sentence_buffer = []
    last_spoken_sign = None

    # ========================
    # Performance controls
    # ========================
    TARGET_FPS = 15
    FRAME_DELAY = 1.0 / TARGET_FPS
    last_frame_time = time.time()

    VOICE_COOLDOWN = 1.0
    last_voice_time = 0

    # ========================
    # Open webcam
    # ========================
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ ERROR: Cannot open webcam!")
        return

    print("✓ Webcam opened")
    print(f"✓ Mode: {mode.upper()}")

    # ========================
    # MediaPipe Hands (FAST)
    # ========================
    with mp_hands.Hands(
        max_num_hands=2,
        model_complexity=0,  # FAST
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as hands:

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # MediaPipe detection
                image, results = mediapipe_detection(frame, hands)

                # Core processing
                sign_detected, is_recording, status = sign_recorder.process_results(results)

                hand_visible = status.get("hand_visible", False)
                confidence = status.get("confidence", 0.0)
                sequence_length = status.get("frame_count", 0)

                # ========================
                # Idle short-circuit
                # ========================
                if not hand_visible:
                    webcam_manager.update(
                        frame=image,
                        results=results,
                        sign_detected="",
                        sentence_buffer=sentence_buffer,
                        is_recording=False,
                        sequence_length=0,
                        hand_visible=False,
                        current_mode=mode,
                        current_sign_name=current_sign_name,
                        dtw_distance=None,
                        confidence=0.0
                    )
                    cv2.waitKey(1)
                    continue

                # ========================
                # UI Update
                # ========================
                webcam_manager.update(
                    frame=image,
                    results=results,
                    sign_detected=sign_detected or "",
                    sentence_buffer=sentence_buffer,
                    is_recording=is_recording,
                    sequence_length=sequence_length,
                    hand_visible=hand_visible,
                    current_mode=mode,
                    current_sign_name=current_sign_name,
                    dtw_distance=sign_recorder.last_dtw_distance,
                    confidence=confidence
                )

                # ========================
                # Handle recognized sign
                # ========================
                if sign_detected and sign_detected not in ["Unknown Sign", "No reference signs"]:
                    if sign_detected != last_spoken_sign:
                        sentence_buffer.append(sign_detected)

                        now = time.time()
                        if now - last_voice_time > VOICE_COOLDOWN:
                            voice_output.speak_sign(sign_detected)
                            last_voice_time = now

                        last_spoken_sign = sign_detected
                        print("✓ Sentence:", " ".join(sentence_buffer))

                # Unknown sign feedback
                if sign_detected == "Unknown Sign" and last_spoken_sign != "Unknown Sign":
                    voice_output.speak_unknown()
                    last_spoken_sign = "Unknown Sign"

                # ========================
                # Keyboard controls
                # ========================
                key = cv2.waitKey(1) & 0xFF

                if key == ord("m"):
                    mode = "record" if mode == "recognize" else "recognize"
                    sign_recorder.mode = mode
                    current_sign_name = None
                    last_spoken_sign = None
                    voice_output.reset()
                    print(f"\n✓ Switched to {mode.upper()} mode\n")

                elif key == ord("c"):
                    sentence_buffer.clear()
                    last_spoken_sign = None
                    print("\n🗑 Sentence buffer cleared\n")

                elif key == ord("n"):
                    if mode != "record":
                        print("⚠ Switch to RECORD mode first (press 'm')")
                    else:
                        new_sign = get_sign_name_input()
                        if new_sign:
                            current_sign_name = new_sign
                            sign_recorder.record(current_sign_name)
                            print(f"🎥 Recording '{current_sign_name}'")

                elif key == ord("q"):
                    print("\n🛑 Exiting...")
                    break

                # ========================
                # FPS limiting
                # ========================
                elapsed = time.time() - last_frame_time
                if elapsed < FRAME_DELAY:
                    time.sleep(FRAME_DELAY - elapsed)
                last_frame_time = time.time()

        except KeyboardInterrupt:
            print("\n⚠ Interrupted by user")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            voice_output.cleanup()
            print("✓ Resources released")
            print("✓ Program closed cleanly")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
