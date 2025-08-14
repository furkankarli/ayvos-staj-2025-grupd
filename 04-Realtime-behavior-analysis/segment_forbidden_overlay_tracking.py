import glob
import os

import cv2
import numpy as np
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO

# ---------------- GPU DEVICE SETUP (MPS for M1) ---------------- #
device = (
    torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
)
print(f"Using device: {device}")

# ---------------- STEP 1: DRAW FORBIDDEN ZONE ---------------- #
drawing_points = []
drawing_done = False


def mouse_draw(event, x, y, flags, param):
    drawing_points
    global drawing_done
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing_points.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        drawing_done = True


def get_forbidden_zone(video_path):
    """Open first frame of the video and let the user draw forbidden zone"""
    global drawing_points, drawing_done
    drawing_points = []
    drawing_done = False

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError("Could not read first frame of the video!")

    clone = frame.copy()
    cv2.namedWindow("Draw Forbidden Zone (L-Click=Point, R-Click=Finish)")
    cv2.setMouseCallback(
        "Draw Forbidden Zone (L-Click=Point, R-Click=Finish)", mouse_draw
    )

    while True:
        temp_frame = clone.copy()
        if len(drawing_points) > 1:
            cv2.polylines(
                temp_frame,
                [np.array(drawing_points, dtype=np.int32)],
                True,
                (0, 0, 255),
                2,
            )
        for p in drawing_points:
            cv2.circle(temp_frame, p, 5, (0, 255, 0), -1)

        cv2.imshow("Draw Forbidden Zone (L-Click=Point, R-Click=Finish)", temp_frame)
        if cv2.waitKey(1) & 0xFF == 27 or drawing_done:  # ESC or Right Click to finish
            break

    cv2.destroyWindow("Draw Forbidden Zone (L-Click=Point, R-Click=Finish)")
    return drawing_points


# ---------------- STEP 2: SETUP MODEL & TRACKER ---------------- #
model = YOLO("yolov8n-seg.pt").to(device)  # YOLO segmentation with MPS GPU
tracker = DeepSort(max_age=30)

# ---------------- STEP 3: PROCESS ALL VIDEOS ---------------- #
videos = glob.glob("test_videos/*.mp4")
output_folder = "processed_videos"
os.makedirs(output_folder, exist_ok=True)

if not videos:
    raise FileNotFoundError("No videos found in the given path!")

for video_path in videos:
    print(f"\n=== Processing video: {video_path} ===")

    # Ask user to draw forbidden zone for this video
    forbidden_zone = get_forbidden_zone(video_path)

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    forbidden_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(forbidden_mask, [np.array(forbidden_zone, dtype=np.int32)], 255)

    output_path = os.path.join(
        output_folder, f"processed_{os.path.basename(video_path)}"
    )
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run segmentation model
        results = model(frame, verbose=False)

        detections = []
        masks_per_id = {}

        for result in results:
            if result.masks is not None:
                for box, mask in zip(
                    result.boxes.xyxy.cpu().numpy(), result.masks.data.cpu().numpy()
                ):
                    x1, y1, x2, y2 = box.astype(int)
                    conf = float(result.boxes.conf[0])
                    cls = int(result.boxes.cls[0])
                    if cls != 0:  # Only detect people (class 0 in COCO)
                        continue
                    detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))
                    masks_per_id[(x1, y1, x2, y2)] = mask

        tracks = tracker.update_tracks(detections, frame=frame)

        # Draw forbidden zone
        cv2.polylines(
            frame, [np.array(forbidden_zone, dtype=np.int32)], True, (0, 0, 255), 2
        )

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)

            # Find mask for this track
            mask_found = None
            for (mx1, my1, mx2, my2), mask in masks_per_id.items():
                if abs(mx1 - x1) < 20 and abs(my1 - y1) < 20:  # crude matching
                    mask_found = mask
                    break

            if mask_found is not None:
                mask_bin = (mask_found * 255).astype(np.uint8)
                mask_resized = cv2.resize(mask_bin, (width, height))
                overlap = cv2.bitwise_and(mask_resized, forbidden_mask)
                overlap_ratio = np.sum(overlap > 0) / max(np.sum(mask_resized > 0), 1)

                # Warning levels
                if overlap_ratio > 0.5:
                    color = (0, 0, 255)  # Red for full entry
                    warning_text = f"ID {track_id}: FULL ENTRY!"
                elif overlap_ratio > 0.1:
                    color = (0, 255, 255)  # Yellow for partial entry
                    warning_text = f"ID {track_id}: PARTIAL ENTRY"
                else:
                    color = (0, 255, 0)  # Green if safe
                    warning_text = f"ID {track_id}: SAFE"

                # Overlay mask
                colored_mask = np.zeros_like(frame)
                colored_mask[:, :, 1] = mask_resized  # green channel
                frame = cv2.addWeighted(frame, 1, colored_mask, 0.4, 0)

                # Draw bbox & text
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    warning_text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

        out.write(frame)
        cv2.imshow("Processed", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    print(f"Processed and saved: {output_path}")

cv2.destroyAllWindows()
