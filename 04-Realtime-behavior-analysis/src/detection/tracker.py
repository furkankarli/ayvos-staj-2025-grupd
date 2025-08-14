"""
Human Tracking Module
DeepSORT ile kişi takibi yapısı.
"""

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort


class HumanTracker:
    """Human tracking using DeepSORT algorithm."""

    def __init__(self, config):
        """
        Initialize the human tracker.

        Args:
            config: Configuration object containing tracking settings
        """
        self.config = config
        self.tracker = DeepSort(max_age=config.TRACKING_MAX_DISAPPEARED)
        self.track_history = {}  # Track ID'ye göre geçmiş pozisyonlar

    def update(
        self, detections: List[Dict], frame: Optional[np.ndarray] = None
    ) -> List[Dict]:
        """
        Update tracks with new detections.

        Args:
            detections: List of detections from detector
            frame: Optional frame for visual tracking

        Returns:
            List of active tracks with ID and bbox
        """
        # DeepSORT formatına çevir
        ds_detections = []
        for det in detections:
            x1, y1, w, h = det["bbox"]
            conf = det["conf"]
            ds_detections.append(([x1, y1, w, h], conf, "person"))

        # Tracking güncelle
        tracks = self.tracker.update_tracks(ds_detections, frame=frame)

        results = []
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)

            # Track geçmişini güncelle
            if track_id not in self.track_history:
                self.track_history[track_id] = []

            centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            self.track_history[track_id].append(centroid)

            # Geçmiş pozisyonları sınırla (son 30 frame)
            if len(self.track_history[track_id]) > 30:
                self.track_history[track_id] = self.track_history[track_id][-30:]

            results.append(
                {
                    "id": track_id,
                    "bbox": (x1, y1, x2 - x1, y2 - y1),  # (x, y, w, h) format
                    "centroid": centroid,
                    "age": len(self.track_history[track_id]),
                }
            )

        return results

    def draw_tracks(
        self,
        frame: np.ndarray,
        tracks: List[Dict],
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """
        Draw tracking bounding boxes and track history on the frame.

        Args:
            frame: Input frame
            tracks: List of active tracks
            color: BGR color tuple
            thickness: Line thickness

        Returns:
            Frame with drawn tracks
        """
        frame_copy = frame.copy()

        for track in tracks:
            x, y, w, h = track["bbox"]
            track_id = track["id"]
            age = track["age"]

            # Bounding box çizimi
            cv2.rectangle(frame_copy, (x, y), (x + w, y + h), color, thickness)

            # Track ID ve yaş bilgisi
            label = f"ID: {track_id} (Age: {age})"
            cv2.putText(
                frame_copy,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                thickness,
            )

            # Track geçmişi çizimi
            if track_id in self.track_history and len(self.track_history[track_id]) > 1:
                points = np.array(self.track_history[track_id], dtype=np.int32)
                cv2.polylines(frame_copy, [points], False, (0, 255, 255), 1)

        return frame_copy

    def get_tracking_stats(self, tracks: List[Dict]) -> Dict:
        """
        Get statistics about tracking.

        Args:
            tracks: List of active tracks

        Returns:
            Dictionary containing tracking statistics
        """
        if not tracks:
            return {
                "active_tracks": 0,
                "avg_age": 0.0,
                "total_tracks": len(self.track_history),
            }

        ages = [track["age"] for track in tracks]

        return {
            "active_tracks": len(tracks),
            "avg_age": np.mean(ages) if ages else 0.0,
            "total_tracks": len(self.track_history),
            "min_age": np.min(ages) if ages else 0,
            "max_age": np.max(ages) if ages else 0,
        }

    def get_track_history(self, track_id: int) -> List[Tuple[int, int]]:
        """
        Get position history for a specific track.

        Args:
            track_id: Track ID

        Returns:
            List of (x, y) positions
        """
        return self.track_history.get(track_id, [])

    def clear_old_tracks(self, max_age: int = 100):
        """
        Clear old tracks from history.

        Args:
            max_age: Maximum age to keep in history
        """
        tracks_to_remove = []
        for track_id, history in self.track_history.items():
            if len(history) > max_age:
                tracks_to_remove.append(track_id)

        for track_id in tracks_to_remove:
            del self.track_history[track_id]
