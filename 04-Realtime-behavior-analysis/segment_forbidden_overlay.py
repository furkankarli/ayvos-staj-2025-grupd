import glob
import os

import cv2
import numpy as np
from ultralytics import YOLO

# ---------------- CONFIGURATION ---------------- #

# Load YOLO segmentation model (You can use yolov8n-seg.pt
# for speed or yolov8m-seg.pt for better accuracy)
model = YOLO("yolov8n-seg.pt")

# Forbidden zone coordinates (example: rectangle polygon)
forbidden_zone = [(100, 200), (400, 200), (400, 500), (100, 500)]

# Create binary mask for forbidden zone
forbidden_mask = np.zeros(
    (720, 1280), dtype=np.uint8
)  # Change resolution according to your videos
cv2.fillPoly(forbidden_mask, [np.array(forbidden_zone, dtype=np.int32)], 255)

# Output folder for processed videos
output_folder = "processed_videos"
os.makedirs(output_folder, exist_ok=True)

# ---------------- PROCESS ALL VIDEOS ---------------- #

# Get all mp4 files in the current folder
videos = glob.glob("test_videos/*.mp4")

for video_path in videos:
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Update forbidden mask size if needed
    forbidden_mask_resized = cv2.resize(forbidden_mask, (width, height))

    # Video writer for saving output
    output_path = os.path.join(
        output_folder, f"processed_{os.path.basename(video_path)}"
    )
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Run segmentation model on the frame
        results = model(frame, verbose=False)

        # Draw forbidden zone on the frame
        cv2.polylines(
            frame, [np.array(forbidden_zone, dtype=np.int32)], True, (0, 0, 255), 2
        )

        warning_triggered = False  # Flag to check if a warning should be displayed

        # Process each detected object
        for result in results:
            if result.masks is not None:
                masks = result.masks.data.cpu().numpy()
                for mask in masks:
                    # Convert mask to binary format
                    mask_bin = (mask * 255).astype(np.uint8)
                    mask_resized = cv2.resize(mask_bin, (width, height))

                    # Check overlap between person mask and forbidden zone
                    overlap = cv2.bitwise_and(mask_resized, forbidden_mask_resized)
                    if np.any(overlap):
                        warning_triggered = True

                    # Draw the segmentation mask on the frame (transparent overlay)
                    colored_mask = np.zeros_like(frame)
                    colored_mask[:, :, 1] = mask_resized  # Green channel
                    frame = cv2.addWeighted(frame, 1, colored_mask, 0.5, 0)

        # Display warning text if needed
        if warning_triggered:
            cv2.putText(
                frame,
                "WARNING: Person in forbidden zone!",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3,
            )

        # Write frame to output video
        out.write(frame)

        # Optional: Show real-time preview
        cv2.imshow("Processed Video", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    print(f"Processed and saved: {output_path}")

cv2.destroyAllWindows()
