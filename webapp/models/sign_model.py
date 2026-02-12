import numpy as np


class SignModel:
    """
    Represents a single sign gesture as a sequence of feature embeddings.

    Optimizations:
    - Uses normalized landmarks
    - Reduces dimensionality (angles + distances)
    - DTW-friendly embeddings
    - CPU-efficient
    """

    def __init__(self, left_hand_seq, right_hand_seq):
        """
        :param left_hand_seq:  list of frames, each (63,)
        :param right_hand_seq: list of frames, each (63,)
        """

        self.has_left_hand = self._is_valid(left_hand_seq)
        self.has_right_hand = self._is_valid(right_hand_seq)

        self.lh_embedding = self._build_embedding(left_hand_seq) if self.has_left_hand else []
        self.rh_embedding = self._build_embedding(right_hand_seq) if self.has_right_hand else []

    # ======================================================================
    # VALIDATION
    # ======================================================================

    def _is_valid(self, seq):
        if seq is None:
            return False
        try:
            if len(seq) == 0:
                return False
        except TypeError:
            return False
        return bool(np.any(np.array(seq)))

    # ======================================================================
    # EMBEDDING PIPELINE
    # ======================================================================

    def _build_embedding(self, seq):
        """
        Converts raw landmark frames into compact feature embeddings.

        Input frame shape : (63,)
        Output frame shape: (~20 features)

        This massively speeds up DTW.
        """

        embeddings = []

        for frame in seq:
            landmarks = np.array(frame, dtype=np.float32).reshape(21, 3)
            features = self._extract_features(landmarks)
            embeddings.append(features)

        return np.array(embeddings, dtype=np.float32)

    # ======================================================================
    # FEATURE EXTRACTION (CORE)
    # ======================================================================

    def _extract_features(self, lm):
        """
        Extract compact, robust features from normalized landmarks.

        Features used:
        - Finger tip distances from wrist
        - Finger joint angles
        - Palm spread metric
        """

        wrist = lm[0]

        # Fingertips: thumb, index, middle, ring, pinky
        tips = lm[[4, 8, 12, 16, 20]]

        # MCP joints
        mcps = lm[[2, 5, 9, 13, 17]]

        # -------------------------
        # Distance features (5)
        # -------------------------
        tip_distances = np.linalg.norm(tips - wrist, axis=1)

        # -------------------------
        # Angle features (5)
        # -------------------------
        angles = []
        for tip, mcp in zip(tips, mcps):
            v1 = tip - mcp
            v2 = mcp - wrist
            angle = self._angle_between(v1, v2)
            angles.append(angle)

        # -------------------------
        # Palm spread (1)
        # -------------------------
        palm_spread = np.mean(
            np.linalg.norm(tips - tips.mean(axis=0), axis=1)
        )

        # Final feature vector (~11 dims)
        return np.concatenate([tip_distances, angles, [palm_spread]])

    # ======================================================================
    # MATH HELPERS
    # ======================================================================

    def _angle_between(self, v1, v2):
        """
        Compute angle between two vectors (safe).
        """
        denom = (np.linalg.norm(v1) * np.linalg.norm(v2)) + 1e-6
        cos_theta = np.dot(v1, v2) / denom
        return np.clip(cos_theta, -1.0, 1.0)
