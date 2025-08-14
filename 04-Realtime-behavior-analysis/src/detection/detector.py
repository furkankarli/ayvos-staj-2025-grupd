"""
Human Detection Module
YOLOv8 ile insan tespiti yapısı.
"""

from typing import Dict, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


class HumanDetector:
    """Human detection using YOLOv8 model."""

    def __init__(self, config):
        """
        Initialize the human detector.

        Args:
            config: Configuration object containing model paths and settings
        """
        self.config = config
        self.model = YOLO(str(config.YOLO_MODEL_PATH))
        self.confidence_threshold = config.DETECTION_CONFIDENCE

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect humans in the given frame.

        Args:
            frame: Input frame as numpy array

        Returns:
            List of detections with bbox and confidence
        """
        results = self.model(frame, imgsz=640, conf=self.confidence_threshold)
        detections = []

        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls != 0:  # Sadece 'person' sınıfı
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])

                # Confidence threshold kontrolü
                if conf < self.confidence_threshold:
                    continue

                detections.append(
                    {
                        "bbox": (x1, y1, x2 - x1, y2 - y1),  # (x, y, w, h) format
                        "conf": conf,
                        "class": "person",
                    }
                )

        return detections

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Dict],
        color: Tuple[int, int, int] = (255, 0, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """
        Draw detection bounding boxes on the frame.

        Args:
            frame: Input frame
            detections: List of detections
            color: BGR color tuple
            thickness: Line thickness

        Returns:
            Frame with drawn detections
        """
        frame_copy = frame.copy()

        for det in detections:
            x, y, w, h = det["bbox"]
            conf = det["conf"]

            # Bounding box çizimi
            cv2.rectangle(frame_copy, (x, y), (x + w, y + h), color, thickness)

            # Confidence score yazısı
            label = f"Person: {conf:.2f}"
            cv2.putText(
                frame_copy,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                thickness,
            )

        return frame_copy

    def get_detection_stats(self, detections: List[Dict]) -> Dict:
        """
        Get statistics about detections.

        Args:
            detections: List of detections

        Returns:
            Dictionary containing detection statistics
        """
        if not detections:
            return {
                "count": 0,
                "avg_confidence": 0.0,
                "min_confidence": 0.0,
                "max_confidence": 0.0,
            }

        confidences = [det["conf"] for det in detections]

        return {
            "count": len(detections),
            "avg_confidence": np.mean(confidences),
            "min_confidence": np.min(confidences),
            "max_confidence": np.max(confidences),
        }
