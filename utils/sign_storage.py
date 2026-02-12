import os
import pickle
from datetime import datetime
from pathlib import Path
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SIGNS_DIR = str(PROJECT_ROOT / "data" / "signs")


def _ensure_signs_directory():
    if not os.path.exists(SIGNS_DIR):
        os.makedirs(SIGNS_DIR, exist_ok=True)


def _append_sequence(sequences, sign_name, left_hand, right_hand):
    if sign_name not in sequences:
        sequences[sign_name] = []
    sequences[sign_name].append(
        (np.array(left_hand, dtype=np.float32), np.array(right_hand, dtype=np.float32))
    )


def save_sign_sequence(sign_name, left_hand_list, right_hand_list):
    """
    Save a sign sequence to disk.

    Stored as .npy in data/signs/<sign_name>/sequence_<timestamp>.npy.
    """
    _ensure_signs_directory()
    sign_dir = os.path.join(SIGNS_DIR, sign_name)
    os.makedirs(sign_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"sequence_{timestamp}.npy"
    filepath = os.path.join(sign_dir, filename)

    payload = {
        "sign_name": sign_name,
        "left_hand": np.array(left_hand_list, dtype=np.float32),
        "right_hand": np.array(right_hand_list, dtype=np.float32),
        "timestamp": timestamp,
    }
    np.save(filepath, payload, allow_pickle=True)
    print(f"Saved sign '{sign_name}' to {filepath}")
    return filepath


def _load_npy_sequence(filepath):
    loaded = np.load(filepath, allow_pickle=True)
    if hasattr(loaded, "item"):
        data = loaded.item()
        if isinstance(data, dict) and "left_hand" in data and "right_hand" in data:
            sign_name = data.get("sign_name")
            return sign_name, data["left_hand"], data["right_hand"]
    return None, None, None


def _load_pkl_sequence(filepath):
    with open(filepath, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "left_hand" in data and "right_hand" in data:
        return data["left_hand"], data["right_hand"]
    return None, None


def load_all_sign_sequences():
    """
    Load all saved sign sequences from disk.

    Supports:
    - New format: data/signs/<sign_name>/sequence_*.npy
    - Legacy format: data/signs/<sign_name>.pkl
    """
    sequences = {}
    if not os.path.exists(SIGNS_DIR):
        return sequences

    for entry in os.listdir(SIGNS_DIR):
        path = os.path.join(SIGNS_DIR, entry)

        if os.path.isdir(path):
            sign_name = entry
            for filename in os.listdir(path):
                file_path = os.path.join(path, filename)
                if filename.endswith(".npy"):
                    _, left_hand, right_hand = _load_npy_sequence(file_path)
                    if left_hand is not None and right_hand is not None:
                        _append_sequence(sequences, sign_name, left_hand, right_hand)
                elif filename.endswith(".pkl"):
                    left_hand, right_hand = _load_pkl_sequence(file_path)
                    if left_hand is not None and right_hand is not None:
                        _append_sequence(sequences, sign_name, left_hand, right_hand)
            continue

        if entry.endswith(".pkl"):
            sign_name = entry[:-4]
            left_hand, right_hand = _load_pkl_sequence(path)
            if left_hand is not None and right_hand is not None:
                _append_sequence(sequences, sign_name, left_hand, right_hand)
        elif entry.endswith(".npy"):
            sign_name_from_file, left_hand, right_hand = _load_npy_sequence(path)
            sign_name = sign_name_from_file or entry[:-4]
            if left_hand is not None and right_hand is not None:
                _append_sequence(sequences, sign_name, left_hand, right_hand)

    return sequences


def get_available_signs():
    """
    Get list of available sign names.
    """
    return sorted(load_all_sign_sequences().keys())
