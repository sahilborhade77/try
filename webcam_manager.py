import cv2
import numpy as np
import mediapipe as mp


WHITE_COLOR = (245, 242, 226)
RED_COLOR = (25, 35, 240)
GREEN_COLOR = (25, 200, 25)
YELLOW_COLOR = (25, 200, 200)
CYAN_COLOR = (200, 200, 25)

HEIGHT = 600


class WebcamManager(object):
    """Object that displays the Webcam output, draws the landmarks detected and
    outputs the sign prediction
    """

    def __init__(self):
        self.sign_detected = ""

    def update(
        self, frame: np.ndarray, results, sign_detected: str, sentence_buffer: list = None,
        is_recording: bool = False, sequence_length: int = 0, hand_visible: bool = False,
        current_mode: str = "recognize", current_sign_name: str = None,
        dtw_distance: float = None, confidence: float = 0.0
    ):
        """
        Update the webcam display with landmarks, text, and comprehensive status information.
        
        :param frame: Current webcam frame
        :param results: MediaPipe detection results
        :param sign_detected: Currently detected sign
        :param sentence_buffer: List of recognized signs (NEW)
        :param is_recording: Whether currently recording
        :param sequence_length: Number of frames collected
        :param hand_visible: Whether hand is visible (NEW)
        :param current_mode: Current mode ("record" or "recognize")
        :param current_sign_name: Name of sign being recorded
        :param dtw_distance: DTW distance for debugging
        :param confidence: Prediction confidence (NEW)
        """
        self.sign_detected = sign_detected if sign_detected else (
            " ".join(sentence_buffer) if sentence_buffer else ""
        )

        # Draw landmarks
        self.draw_landmarks(frame, results)

        WIDTH = int(HEIGHT * len(frame[0]) / len(frame))
        # Resize frame
        frame = cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)

        # Flip the image vertically for mirror effect
        frame = cv2.flip(frame, 1)

        # Draw mode indicator
        mode_text = f"MODE: {current_mode.upper()}"
        mode_color = RED_COLOR if current_mode == "record" else GREEN_COLOR
        cv2.putText(
            frame,
            mode_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            mode_color,
            2,
        )

        # ===== NEW: Gesture-based control instructions =====
        if current_mode == "recognize" and not is_recording:
            gesture_text = "🖐️  Open Palm to Record"
            cv2.putText(
                frame,
                gesture_text,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                YELLOW_COLOR,
                1,
            )
        elif is_recording:
            gesture_text = "✊ Make a Fist to Stop"
            cv2.putText(
                frame,
                gesture_text,
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                YELLOW_COLOR,
                1,
            )

        # ===== NEW: Hand visibility status =====
        hand_status = "🟢 HANDS VISIBLE" if hand_visible else "🔴 NO HANDS"
        hand_color = GREEN_COLOR if hand_visible else RED_COLOR
        cv2.putText(
            frame,
            hand_status,
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            hand_color,
            2,
        )

        # Draw recording status message with frame counter
        if is_recording:
            status_text = f"🎥 RECORDING... ({sequence_length}/45 frames)"
            cv2.putText(
                frame,
                status_text,
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                RED_COLOR,
                2,
            )
            
            # Draw recording progress bar
            bar_width = 200
            bar_height = 20
            bar_x, bar_y = 10, 140
            
            # Background
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), WHITE_COLOR, -1)
            
            # Progress
            progress = min(int(bar_width * sequence_length / 45), bar_width)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + progress, bar_y + bar_height), RED_COLOR, -1)
            
            # Border
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), 2)
        
        # Draw current sign name being recorded
        if current_sign_name and current_mode == "record":
            sign_name_text = f"Recording: '{current_sign_name}'"
            cv2.putText(
                frame,
                sign_name_text,
                (10, 175),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                YELLOW_COLOR,
                2,
            )

        # ===== NEW: Confidence bar (for recognized signs) =====
        if confidence > 0 and not is_recording:
            conf_text = f"Confidence: {confidence:.1%}"
            cv2.putText(
                frame,
                conf_text,
                (10, 175),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                CYAN_COLOR,
                1,
            )
            
            # Confidence bar
            bar_width = 150
            bar_height = 15
            bar_x, bar_y = 180, 165
            
            # Background
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), -1)
            
            # Confidence fill
            conf_fill = int(bar_width * confidence)
            conf_color = GREEN_COLOR if confidence > 0.7 else YELLOW_COLOR if confidence > 0.5 else RED_COLOR
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + conf_fill, bar_y + bar_height), conf_color, -1)
            
            # Border
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), WHITE_COLOR, 1)

        # Draw DTW distance (for debugging)
        if dtw_distance is not None:
            distance_text = f"DTW Distance: {dtw_distance:.2f}"
            cv2.putText(
                frame,
                distance_text,
                (10, 205),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                CYAN_COLOR,
                1,
            )

        # ===== NEW: Sentence buffer display =====
        if sentence_buffer:
            sentence_text = "Sentence: " + " ".join(sentence_buffer)
            # Use draw_text for better background
            frame = self.draw_sentence_buffer(frame, sentence_text)
        else:
            # Draw current sign detection if no sentence
            frame = self.draw_text(frame)

        # Status circle color
        color = WHITE_COLOR
        if is_recording:
            color = RED_COLOR

        # Draw status indicator (circle)
        cv2.circle(frame, (WIDTH - 30, 30), 15, color, -1)
        
        cv2.imshow("OpenCV Feed", frame)

    def draw_sentence_buffer(self, frame, sentence_text, font=cv2.FONT_HERSHEY_COMPLEX,
                           font_size=1.2, font_thickness=2, offset=int(HEIGHT * 0.02)):
        """
        ===== NEW: Draw sentence buffer with nice background
        
        :param frame: The frame to draw on
        :param sentence_text: The sentence to display
        :param font: Font type
        :param font_size: Font size
        :param font_thickness: Font thickness
        :param offset: Offset from bottom
        :return: Updated frame
        """
        window_w = int(HEIGHT * len(frame[0]) / len(frame))
        
        (text_w, text_h), _ = cv2.getTextSize(
            sentence_text, font, font_size, font_thickness
        )
        
        text_x, text_y = int((window_w - text_w) / 2), HEIGHT - text_h - offset
        
        # Draw background rectangle
        cv2.rectangle(frame, (0, text_y - offset), (window_w, HEIGHT), (100, 150, 100), -1)
        
        # Draw text
        cv2.putText(
            frame,
            sentence_text,
            (text_x, text_y + text_h + font_size - 1),
            font,
            font_size,
            (255, 255, 255),  # White text
            font_thickness,
        )
        return frame


        (text_w, text_h), _ = cv2.getTextSize(
            self.sign_detected, font, font_size, font_thickness
        )

        text_x, text_y = int((window_w - text_w) / 2), HEIGHT - text_h - offset

        cv2.rectangle(frame, (0, text_y - offset), (window_w, HEIGHT), bg_color, -1)
        cv2.putText(
            frame,
            self.sign_detected,
            (text_x, text_y + text_h + font_size - 1),
            font,
            font_size,
            (118, 62, 37),
            font_thickness,
        )
        return frame

    @staticmethod
    def draw_landmarks(image, results):
        mp_holistic = mp.solutions.holistic  # Holistic model
        mp_drawing = mp.solutions.drawing_utils  # Drawing utilities

        # Draw left hand connections
        mp_drawing.draw_landmarks(
            image,
            landmark_list=results.left_hand_landmarks,
            connections=mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(
                color=(232, 254, 255), thickness=1, circle_radius=1
            ),
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=(255, 249, 161), thickness=2, circle_radius=2
            ),
        )
        # Draw right hand connections
        mp_drawing.draw_landmarks(
            image,
            landmark_list=results.right_hand_landmarks,
            connections=mp_holistic.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(
                color=(232, 254, 255), thickness=1, circle_radius=2
            ),
            connection_drawing_spec=mp_drawing.DrawingSpec(
                color=(255, 249, 161), thickness=2, circle_radius=2
            ),
        )
