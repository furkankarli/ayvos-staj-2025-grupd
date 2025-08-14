import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import cv2

# --- src klasörünü Python modül yolu olarak ekle ---
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if src_path not in sys.path:
    sys.path.append(src_path)

# --- Detection, Tracking ve Pose import ---
from detection.detector import HumanDetector
from detection.tracker import HumanTracker
from pose.pose_estimator import PoseEstimator  # noqa: E402


# --- Config sınıfı ---
class Config:
    YOLO_MODEL_PATH = Path(
        r"D:\PycharmProjects\Staj\ayvos-staj-2025-grupd\models\yolov8n.pt"
    )
    POSE_MODEL_PATH = Path(
        r"D:\PycharmProjects\Staj\ayvos-staj-2025-grupd\models\yolo11n-pose.pt"
    )
    MODELS_DIR = Path(r"D:\PycharmProjects\Staj\ayvos-staj-2025-grupd\models")
    DETECTION_CONFIDENCE = 0.25
    TRACKING_MAX_DISAPPEARED = 30
    POSE_CONFIDENCE = 0.3
    KEYPOINT_CONFIDENCE = 0.5


# --- Log ve zaman takibi ---
class SafetyMonitor:
    def __init__(self, log_dir: str):
        self.track_start_time: Dict[int, float] = {}
        self.last_seen_time: Dict[int, float] = {}
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

    def update_times(self, tracks: List[Dict]):
        current_time = time.time()
        for track in tracks:
            track_id = track["id"]
            if track_id not in self.track_start_time:
                self.track_start_time[track_id] = current_time
            self.last_seen_time[track_id] = current_time

    def check_inactive(self, threshold_seconds: int = 10) -> List[int]:
        current_time = time.time()
        inactive = [
            track_id
            for track_id, last_time in self.last_seen_time.items()
            if current_time - last_time > threshold_seconds
        ]
        return inactive

    def save_logs(self, pose_results: List[Dict]):
        csv_path = os.path.join(self.log_dir, "log.csv")
        json_path = os.path.join(self.log_dir, "log.json")

        # CSV kaydet
        with open(csv_path, mode="w", newline="") as csvfile:
            fieldnames = [
                "track_id",
                "start_time",
                "last_seen",
                "duration_sec",
                "pose_class",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for track_id in self.track_start_time:
                start = self.track_start_time[track_id]
                last = self.last_seen_time.get(track_id, start)
                duration = last - start
                # Poz bilgisi ekle
                pose_class = next(
                    (
                        p["pose_class"]
                        for p in pose_results
                        if p["track_id"] == track_id
                    ),
                    "unknown",
                )
                writer.writerow(
                    {
                        "track_id": track_id,
                        "start_time": start,
                        "last_seen": last,
                        "duration_sec": duration,
                        "pose_class": pose_class,
                    }
                )

        # JSON kaydet
        data = []
        for track_id in self.track_start_time:
            start = self.track_start_time[track_id]
            last = self.last_seen_time.get(track_id, start)
            duration = last - start
            pose_class = next(
                (p["pose_class"] for p in pose_results if p["track_id"] == track_id),
                "unknown",
            )
            data.append(
                {
                    "track_id": track_id,
                    "start_time": start,
                    "last_seen": last,
                    "duration_sec": duration,
                    "pose_class": pose_class,
                }
            )
        with open(json_path, "w") as f:
            json.dump(data, f, indent=4)


# --- Main döngü ---
def main():
    input_video_dir = Path(
        r"D:\PycharmProjects\Staj\ayvos-staj-2025-grupd\04-Realtime-behavior-analysis\data\inputs"
    )
    output_video_dir = Path(
        r"D:\PycharmProjects\Staj\ayvos-staj-2025-grupd\04-Realtime-behavior-analysis\data\outputs"
    )
    log_dir = Path(
        r"D:\PycharmProjects\Staj\ayvos-staj-2025-grupd\04-Realtime-behavior-analysis\data\logs"
    )

    os.makedirs(output_video_dir, exist_ok=True)
    config = Config()

    detector = HumanDetector(config=config)
    tracker = HumanTracker(config=config)
    monitor = SafetyMonitor(log_dir)
    pose_estimator = PoseEstimator(config)

    for video_file in input_video_dir.iterdir():
        if not video_file.suffix.lower() in [".mp4", ".avi", ".mkv"]:
            continue

        cap = cv2.VideoCapture(str(video_file))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_path = output_video_dir / video_file.name
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            detections = detector.detect(frame)
            tracks = tracker.update(detections, frame)
            monitor.update_times(tracks)

            # Pose tahmini ve sınıflandırma
            pose_results = pose_estimator.estimate_pose(frame, tracks)
            frame = pose_estimator.draw_keypoints(frame, pose_results)

            # Inactive kontroller
            inactive = monitor.check_inactive(threshold_seconds=10)
            if inactive:
                print(f"Hareketsiz kişi ID'leri: {inactive}")

            out.write(frame)
            cv2.imshow("Analysis", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        out.release()

    monitor.save_logs(pose_results)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
