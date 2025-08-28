import csv

import numpy as np
from scipy.interpolate import interp1d


def interpolate_bounding_boxes(data):
    """
    Interpolates missing bounding box data for vehicles and license plates
    across consecutive video frames.
    """

    # Extract required data from input CSV
    frame_indices = np.array([int(row["frame_idx"]) for row in data])
    car_ids = np.array([int(float(row["car_id"])) for row in data])
    car_bboxes = np.array(
        [list(map(float, row["car_bbox"][1:-1].split())) for row in data]
    )
    plate_bboxes = np.array(
        [list(map(float, row["license_plate_bbox"][1:-1].split())) for row in data]
    )

    interpolated_data = []
    unique_car_ids = np.unique(car_ids)

    # Process each unique car
    for car_id in unique_car_ids:
        frame_indices_for_car = [
            p["frame_idx"]
            for p in data
            if int(float(p["car_id"])) == int(float(car_id))
        ]
        print(frame_indices_for_car, car_id)

        # Filter rows for this specific car ID
        car_mask = car_ids == car_id
        car_frames = frame_indices[car_mask]
        car_bboxes_interpolated = []
        plate_bboxes_interpolated = []

        first_frame = car_frames[0]

        # Iterate through bounding boxes of the car
        for i in range(len(car_bboxes[car_mask])):
            frame_num = car_frames[i]
            car_bbox = car_bboxes[car_mask][i]
            plate_bbox = plate_bboxes[car_mask][i]

            if i > 0:
                prev_frame_num = car_frames[i - 1]
                prev_car_bbox = car_bboxes_interpolated[-1]
                prev_plate_bbox = plate_bboxes_interpolated[-1]

                # If frames are missing, interpolate bounding boxes linearly
                if frame_num - prev_frame_num > 1:
                    gap = frame_num - prev_frame_num
                    x = np.array([prev_frame_num, frame_num])
                    x_new = np.linspace(
                        prev_frame_num, frame_num, num=gap, endpoint=False
                    )

                    # Interpolate car bounding boxes
                    interp_func = interp1d(
                        x, np.vstack((prev_car_bbox, car_bbox)), axis=0, kind="linear"
                    )
                    interpolated_car_bboxes = interp_func(x_new)

                    # Interpolate license plate bounding boxes
                    interp_func = interp1d(
                        x,
                        np.vstack((prev_plate_bbox, plate_bbox)),
                        axis=0,
                        kind="linear",
                    )
                    interpolated_plate_bboxes = interp_func(x_new)

                    car_bboxes_interpolated.extend(interpolated_car_bboxes[1:])
                    plate_bboxes_interpolated.extend(interpolated_plate_bboxes[1:])

            # Append current frame bounding boxes
            car_bboxes_interpolated.append(car_bbox)
            plate_bboxes_interpolated.append(plate_bbox)

        # Create rows with interpolated values
        for i in range(len(car_bboxes_interpolated)):
            frame_num = first_frame + i
            row = {}
            row["frame_idx"] = str(frame_num)
            row["car_id"] = str(car_id)
            row["car_bbox"] = " ".join(map(str, car_bboxes_interpolated[i]))
            row["license_plate_bbox"] = " ".join(map(str, plate_bboxes_interpolated[i]))

            if str(frame_num) not in frame_indices_for_car:
                # Interpolated row → set text-related fields to '0'
                row["license_plate_bbox_score"] = "0"
                row["license_number"] = "0"
                row["license_number_score"] = "0"
            else:
                # Original row → copy values from input data if available
                original_row = [
                    p
                    for p in data
                    if int(p["frame_idx"]) == frame_num
                    and int(float(p["car_id"])) == int(float(car_id))
                ][0]
                row["license_plate_bbox_score"] = original_row.get(
                    "license_plate_bbox_score", "0"
                )
                row["license_number"] = original_row.get("license_number", "0")
                row["license_number_score"] = original_row.get(
                    "license_number_score", "0"
                )

            interpolated_data.append(row)

    return interpolated_data


# ----------------------------
# Load the CSV file
# ----------------------------
with open("test.csv", "r") as file:
    reader = csv.DictReader(file)
    data = list(reader)

# Perform interpolation
interpolated_data = interpolate_bounding_boxes(data)

# ----------------------------
# Save interpolated results
# ----------------------------
header = [
    "frame_idx",
    "car_id",
    "car_bbox",
    "license_plate_bbox",
    "license_plate_bbox_score",
    "license_number",
    "license_number_score",
]

with open("test_interpolated.csv", "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=header)
    writer.writeheader()
    writer.writerows(interpolated_data)
