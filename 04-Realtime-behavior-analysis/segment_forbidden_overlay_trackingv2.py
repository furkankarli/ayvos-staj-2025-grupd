import glob
import os

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ==================================================
# 1) DEVICE SELECTION (MPS for Macbook GPU acceleration)
# ==================================================
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"[INFO] Using device: {device}")

# ==================================================
# 2) LOAD YOLOv8n-seg MODEL (fast human segmentation)
# ==================================================
model = YOLO("yolov8n-seg.pt").to(device)

# ==================================================
# 3) VIDEO FOLDER SETUP
# ==================================================
video_folder = "test_videos"
output_folder = "processed_videosv2"
os.makedirs(output_folder, exist_ok=True)

# ==================================================
# 4) Polygon Drawing for Forbidden Zone
# ==================================================
drawing = False
polygon_points = []
temp_frame = None


def draw_polygon(event, x, y, flags, param):
    """Mouse callback for drawing forbidden zone polygon."""
    polygon_points, temp_frame
    if event == cv2.EVENT_LBUTTONDOWN:
        polygon_points.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(polygon_points) > 2:
            cv2.fillPoly(temp_frame, [np.array(polygon_points)], (0, 0, 255))
            cv2.imshow("Draw Forbidden Zone", temp_frame)


def create_forbidden_mask(frame):
    """Allow user to draw forbidden zone mask before each video."""
    global polygon_points, temp_frame
    polygon_points = []  # reset points for each video
    temp_frame = frame.copy()
    cv2.namedWindow("Draw Forbidden Zone")
    cv2.setMouseCallback("Draw Forbidden Zone", draw_polygon)

    print(
        "[INFO] Left click: add points | Right click: preview fill "
        "| 's': save | 'q': cancel"
    )

    while True:
        preview = temp_frame.copy()
        if len(polygon_points) > 1:
            cv2.polylines(preview, [np.array(polygon_points)], True, (0, 0, 255), 2)
        cv2.imshow("Draw Forbidden Zone", preview)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s") and len(polygon_points) > 2:
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [np.array(polygon_points)], 255)
            cv2.destroyWindow("Draw Forbidden Zone")
            return mask
        elif key == ord("q"):
            cv2.destroyWindow("Draw Forbidden Zone")
            return None


# ==================================================
# 5) VIDEO PROCESSING LOOP
# ==================================================
videos = glob.glob(os.path.join(video_folder, "*.mp4"))

for video_path in videos:
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print(f"[WARNING] Could not read video: {video_path}")
        continue

    # Ask user to draw forbidden zone before each video
    forbidden_mask = create_forbidden_mask(frame)
    if forbidden_mask is None:
        print("[INFO] Forbidden zone selection skipped for this video.")
        cap.release()
        continue

    # Resize mask to match video resolution
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    resized_mask = cv2.resize(
        forbidden_mask, (width, height), interpolation=cv2.INTER_NEAREST
    )

    # Setup output video writer
    out_path = os.path.join(output_folder, os.path.basename(video_path))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    print(f"[INFO] Processing video: {video_path}")

    # Reset to first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO segmentation
        results = model(frame, verbose=False)

        # Draw forbidden zone in red overlay
        frame[resized_mask > 0] = (
            0.4 * frame[resized_mask > 0] + 0.6 * np.array([0, 0, 255])
        ).astype(np.uint8)

        # Check overlap between person masks and forbidden zone
        for result in results:
            if result.masks is not None:
                for mask in result.masks.data:
                    mask = mask.cpu().numpy().astype(np.uint8) * 255
                    mask_resized = cv2.resize(
                        mask, (width, height), interpolation=cv2.INTER_NEAREST
                    )
                    overlap = cv2.bitwise_and(resized_mask, mask_resized)
                    if np.any(overlap > 0):
                        cv2.putText(
                            frame,
                            "WARNING: In Forbidden Zone",
                            (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            2,
                        )

        out.write(frame)
        cv2.imshow("Processed Video", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()

cv2.destroyAllWindows()
print("[INFO] All videos processed.")
