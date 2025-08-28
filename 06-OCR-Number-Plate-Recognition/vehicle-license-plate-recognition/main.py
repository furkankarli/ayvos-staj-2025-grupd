import cv2
import get_car
import numpy as np
import read_license_plate
import write_csv
from sort.sort import Sort
from ultralytics import YOLO

# Dictionary to store detection and recognition results
results = {}

# Initialize SORT tracker for vehicle tracking
tracker = Sort()

# Load YOLO models
vehicle_detector = YOLO("yolov8n.pt")  # COCO-pretrained model for vehicles
plate_detector = YOLO("license_plate_detector.pt")  # Custom model for license plates

# Load input video
cap = cv2.VideoCapture("./sample.mp4")

# Vehicle class IDs in COCO dataset (car=2, motorcycle=3, bus=5, truck=7)
vehicle_classes = [2, 3, 5, 7]

# Frame index
frame_idx = -1
ret = True

# Process video frame by frame
while ret:
    frame_idx += 1
    ret, frame = cap.read()

    if ret:
        results[frame_idx] = {}

        # ----------------------------
        # Step 1: Detect vehicles
        # ----------------------------
        detections = vehicle_detector(frame)[0]
        vehicle_detections = []
        for det in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = det
            if int(class_id) in vehicle_classes:
                vehicle_detections.append([x1, y1, x2, y2, score])

        # ----------------------------
        # Step 2: Track vehicles
        # ----------------------------
        tracked_vehicles = tracker.update(np.asarray(vehicle_detections))

        # ----------------------------
        # Step 3: Detect license plates
        # ----------------------------
        plates = plate_detector(frame)[0]
        for plate in plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = plate

            # Assign license plate to the nearest tracked vehicle
            car_x1, car_y1, car_x2, car_y2, car_id = get_car(plate, tracked_vehicles)

            if car_id != -1:
                # ----------------------------
                # Step 4: Crop license plate
                # ----------------------------
                plate_crop = frame[int(y1) : int(y2), int(x1) : int(x2), :]

                # Preprocess license plate (grayscale + thresholding)
                plate_gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                _, plate_thresh = cv2.threshold(
                    plate_gray, 64, 255, cv2.THRESH_BINARY_INV
                )

                # ----------------------------
                # Step 5: Read license plate text (OCR)
                # ----------------------------
                plate_text, plate_text_score = read_license_plate(plate_thresh)

                # Save results if OCR was successful
                if plate_text is not None:
                    results[frame_idx][car_id] = {
                        "car": {"bbox": [car_x1, car_y1, car_x2, car_y2]},
                        "license_plate": {
                            "bbox": [x1, y1, x2, y2],
                            "text": plate_text,
                            "bbox_score": score,
                            "text_score": plate_text_score,
                        },
                    }

# ----------------------------
# Step 6: Write results to CSV
# ----------------------------
write_csv(results, "./test.csv")
