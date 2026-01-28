import cv2
import os
import numpy as np
import pickle as pkl
import mediapipe as mp
from utils.mediapipe_utils import mediapipe_detection


def landmark_to_array(mp_landmark_list):
    """Return a np array of size (nb_keypoints x 3)"""
    keypoints = []
    for landmark in mp_landmark_list.landmark:
        keypoints.append([landmark.x, landmark.y, landmark.z])
    return np.nan_to_num(keypoints)


def normalize_hand_landmarks(landmarks):
    """
    Normalize hand landmarks for distance-invariant recognition.
    
    Uses the wrist (landmark 0) as origin and scales by hand size.
    This makes the model robust to camera distance variations.
    
    :param landmarks: Array of shape (21, 3) or list of 63 values
    :return: Normalized landmarks as array of shape (21, 3)
    """
    landmarks = np.array(landmarks).reshape((21, 3))
    
    # Check if any landmarks are detected (all zeros = no hand)
    if np.sum(landmarks) == 0:
        return landmarks
    
    # Use wrist (landmark 0) as origin
    wrist = landmarks[0].copy()
    landmarks = landmarks - wrist
    
    # Calculate hand size (max distance between any two landmarks)
    # This is robust and helps scale invariance
    max_distance = 0
    for i in range(len(landmarks)):
        for j in range(i + 1, len(landmarks)):
            dist = np.linalg.norm(landmarks[i] - landmarks[j])
            max_distance = max(max_distance, dist)
    
    # Avoid division by zero
    if max_distance > 0:
        landmarks = landmarks / max_distance
    
    return landmarks


def extract_landmarks(results):
    """Extract the results of both hands and convert them to a np array of size
    if a hand doesn't appear, return an array of zeros
    
    Uses normalized landmarks for distance-invariant recognition.

    :param results: mediapipe object that contains the 3D position of all keypoints
    :return: Two np arrays of size (1, 21 * 3) = (1, nb_keypoints * nb_coordinates) corresponding to both hands
    """
    pose = landmark_to_array(results.pose_landmarks).reshape(99).tolist()

    left_hand = np.zeros(63).tolist()
    if results.left_hand_landmarks:
        # Normalize left hand landmarks for distance invariance
        lh_landmarks = landmark_to_array(results.left_hand_landmarks)
        lh_normalized = normalize_hand_landmarks(lh_landmarks)
        left_hand = lh_normalized.reshape(63).tolist()

    right_hand = np.zeros(63).tolist()
    if results.right_hand_landmarks:
        # Normalize right hand landmarks for distance invariance
        rh_landmarks = landmark_to_array(results.right_hand_landmarks)
        rh_normalized = normalize_hand_landmarks(rh_landmarks)
        right_hand = rh_normalized.reshape(63).tolist()
    
    return pose, left_hand, right_hand


def save_landmarks_from_video(video_name):
    landmark_list = {"pose": [], "left_hand": [], "right_hand": []}
    sign_name = video_name.split("-")[0]

    # Set the Video stream
    cap = cv2.VideoCapture(
        os.path.join("data", "videos", sign_name, video_name + ".mp4")
    )
    with mp.solutions.holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # Make detections
                image, results = mediapipe_detection(frame, holistic)

                # Store results
                pose, left_hand, right_hand = extract_landmarks(results)
                landmark_list["pose"].append(pose)
                landmark_list["left_hand"].append(left_hand)
                landmark_list["right_hand"].append(right_hand)
            else:
                break
        cap.release()

    # Create the folder of the sign if it doesn't exists
    path = os.path.join("data", "dataset", sign_name)
    if not os.path.exists(path):
        os.mkdir(path)

    # Create the folder of the video data if it doesn't exists
    data_path = os.path.join(path, video_name)
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    # Saving the landmark_list in the correct folder
    save_array(
        landmark_list["pose"], os.path.join(data_path, f"pose_{video_name}.pickle")
    )
    save_array(
        landmark_list["left_hand"], os.path.join(data_path, f"lh_{video_name}.pickle")
    )
    save_array(
        landmark_list["right_hand"], os.path.join(data_path, f"rh_{video_name}.pickle")
    )


def save_array(arr, path):
    file = open(path, "wb")
    pkl.dump(arr, file)
    file.close()


def load_array(path):
    file = open(path, "rb")
    arr = pkl.load(file)
    file.close()
    return np.array(arr)
