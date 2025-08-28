import glob
import os

import cv2
import easyocr  # For OCR
import pandas as pd
from ultralytics import YOLO

# ----------------------------
# Load YOLO models
# ----------------------------
vehicle_model = YOLO("yolov8n.pt")  # Pretrained vehicle detection
plate_model = YOLO("license_plate_detector.pt")  # License plate detection

# Initialize OCR reader
reader = easyocr.Reader(["en"])  # Add 'tr' if you want Turkish as well

# ----------------------------
# Define input and output folders
# ----------------------------
input_folder = "dataset_images"
output_folder = "outputs"
os.makedirs(output_folder, exist_ok=True)

results_list = []

# Get all image paths
image_paths = glob.glob(os.path.join(input_folder, "*.jpg")) + glob.glob(
    os.path.join(input_folder, "*.png")
)

# ----------------------------
# Process each image
# ----------------------------
for img_path in image_paths:
    img = cv2.imread(img_path)

    # 1. Detect vehicles (optional)
    vehicles = vehicle_model(img)[0]

    # 2. Detect license plates
    plates = plate_model(img)[0]

    annotated_img = img.copy()

    for box in plates.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        plate_crop = img[y1:y2, x1:x2]

        # ----------------------------
        # OCR to read license plate
        # ----------------------------
        text = ""
        if plate_crop.size != 0:
            ocr_result = reader.readtext(plate_crop)
            if len(ocr_result) > 0:
                text = ocr_result[0][-2]  # Extract recognized text

        # ----------------------------
        # Draw bounding box and text
        # ----------------------------
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        if text != "":
            cv2.putText(
                annotated_img,
                text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

        # ----------------------------
        # Save results for CSV
        # ----------------------------
        results_list.append(
            {
                "image": os.path.basename(img_path),
                "plate_bbox": [x1, y1, x2, y2],
                "plate_text": text,
            }
        )

    # ----------------------------
    # Save annotated image
    # ----------------------------
    save_path = os.path.join(output_folder, os.path.basename(img_path))
    cv2.imwrite(save_path, annotated_img)

# ----------------------------
# Save CSV results
# ----------------------------
df = pd.DataFrame(results_list)
df.to_csv("results_images.csv", index=False)

print(
    "✅ Done! Annotated images are saved in 'outputs/', results in 'results_images.csv'."
)
