# test_detection_tracking.py
import cv2
from config import Config
from detection.detector import HumanDetector
from detection.tracker import HumanTracker


def main():
    # Model ve tracker başlat
    detector = HumanDetector(config=Config)
    tracker = HumanTracker(config=Config)

    # Test videosu
    video_path = r"D:\PycharmProjects\Staj\ayvos-staj-2025-grupd\04-Realtime-behavior-analysis\data\inputs\media4.mp4"
    cap = cv2.VideoCapture(video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # İnsan tespiti
        detections = detector.detect(frame)

        # Takip güncelle
        tracks = tracker.update(detections, frame=frame)

        # Görsel çizimler
        frame = detector.draw_detections(frame, detections)
        frame = tracker.draw_tracks(frame, tracks)

        # Pencereye göster
        cv2.imshow("Human Detection + Tracking", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
