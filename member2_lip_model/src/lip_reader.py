
import torch
import numpy as np
import cv2
from typing import List, Dict, Tuple
from collections import deque, Counter

class LipReadingModel:
    """Lip Reading Model - Member 2 Deliverable"""

    def __init__(self, vocabulary_limit: int = 50):
        self.sequence_length = 16
        self.confidence_threshold = 0.7
        self.vocabulary = [
            "hello", "yes", "no", "help", "please", "thank", "you",
            "water", "food", "good", "bad", "stop", "go", "come",
            "what", "where", "when", "who", "how", "why",
            "want", "need", "have", "give", "take", "see",
            "hear", "speak", "listen", "understand", "know",
            "love", "like", "sorry", "okay", "fine",
            "tired", "happy", "sad", "angry", "scared",
            "doctor", "hospital", "emergency", "pain", "medicine",
            "family", "friend", "name", "my", "is"
        ][:vocabulary_limit]

        self.buffer = deque(maxlen=self.sequence_length)
        self.history = deque(maxlen=3)

    def add_frame(self, lip_crop: np.ndarray) -> bool:
        """Add frame from Member 1 (128x64 grayscale)"""
        if lip_crop is None:
            return False
        if lip_crop.shape != (64, 128):
            lip_crop = cv2.resize(lip_crop, (128, 64))
        if len(lip_crop.shape) == 3:
            lip_crop = cv2.cvtColor(lip_crop, cv2.COLOR_BGR2GRAY)
        self.buffer.append(lip_crop)
        return len(self.buffer) >= self.sequence_length

    def predict(self) -> Dict:
        """Main prediction function"""
        if len(self.buffer) < self.sequence_length:
            return {"text": "", "confidence": 0.0, "is_valid": False}

        frames = list(self.buffer)

        # Mock prediction (replace with actual model)
        text, conf = self._mock_predict(frames)

        # Vocabulary constraint
        if text not in self.vocabulary:
            text, conf = "", conf * 0.3

        # Threshold
        is_valid = conf >= self.confidence_threshold
        if not is_valid:
            text = ""

        # Smoothing
        if is_valid:
            self.history.append(text)
            text = Counter(self.history).most_common(1)[0][0]

        return {
            "text": text,
            "confidence": conf if is_valid else 0.0,
            "is_valid": is_valid,
            "raw_text": text,
            "raw_confidence": conf
        }

    def _mock_predict(self, frames: List[np.ndarray]) -> Tuple[str, float]:
        """Placeholder for actual model inference"""
        import random
        avg_var = np.mean([np.var(f) for f in frames])
        if avg_var > 0.05:
            return ("hello", 0.85)
        return ("yes", 0.90)

def predict_from_lips(lip_frames: List[np.ndarray]) -> Tuple[str, float]:
    """One-shot prediction for Member 3"""
    model = LipReadingModel()
    for frame in lip_frames:
        model.add_frame(frame)
    result = model.predict()
    return (result["text"], result["confidence"])
