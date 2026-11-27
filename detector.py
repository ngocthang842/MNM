"""
Wrapper phát hiện khuôn mặt dùng YuNet (thread-safe).
Sau này thay model: chỉ cần giữ format trả về {x, y, w, h, score, landmarks}.
"""
import threading
from pathlib import Path

import cv2
import numpy as np

MODEL_PATH = Path(__file__).parent / "models" / "face_detection_yunet_2023mar.onnx"


class FaceDetector:
    def __init__(self, model_path: str = str(MODEL_PATH), score_threshold: float = 0.6):
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Không tìm thấy model: {model_path}\n"
                f"Tải tại: https://github.com/opencv/opencv_zoo/raw/main/models/"
                f"face_detection_yunet/face_detection_yunet_2023mar.onnx"
            )
        self._lock = threading.Lock()
        self.detector = cv2.FaceDetectorYN.create(
            model_path, "", (320, 320),
            score_threshold=score_threshold,
            nms_threshold=0.3,
            top_k=5000,
        )

    def detect(self, image_bgr: np.ndarray):
        h, w = image_bgr.shape[:2]
        with self._lock:
            self.detector.setInputSize((w, h))
            _, faces = self.detector.detect(image_bgr)
        results = []
        if faces is None:
            return results
        for f in faces:
            x, y, fw, fh = [int(v) for v in f[:4]]
            score = float(f[-1])
            landmarks = f[4:14].reshape(5, 2).tolist()
            results.append({
                "x": max(0, x), "y": max(0, y),
                "w": fw, "h": fh,
                "score": score,
                "landmarks": landmarks,
            })
        return results


_detector = None
_init_lock = threading.Lock()


def get_detector() -> FaceDetector:
    global _detector
    if _detector is None:
        with _init_lock:
            if _detector is None:
                _detector = FaceDetector()
    return _detector
