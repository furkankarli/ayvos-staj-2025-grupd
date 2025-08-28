import string

import easyocr

# Initialize EasyOCR reader (English, CPU mode)
reader = easyocr.Reader(["en"], gpu=False)

# Character conversion dictionaries (to handle OCR misclassifications)
dict_char_to_int = {"O": "0", "I": "1", "J": "3", "A": "4", "G": "6", "S": "5"}

dict_int_to_char = {"0": "O", "1": "I", "3": "J", "4": "A", "6": "G", "5": "S"}


def write_csv(results, output_path):
    """
    Write detection and recognition results to a CSV file.

    Args:
        results (dict): Dictionary containing frame-wise detection results.
        output_path (str): Path where the CSV file will be saved.
    """
    with open(output_path, "w") as f:
        f.write(
            "{},{},{},{},{},{},{}\n".format(
                "frame_idx",
                "car_id",
                "car_bbox",
                "license_plate_bbox",
                "license_plate_bbox_score",
                "license_number",
                "license_number_score",
            )
        )

        for frame_idx in results.keys():
            for car_id in results[frame_idx].keys():
                if (
                    "car" in results[frame_idx][car_id].keys()
                    and "license_plate" in results[frame_idx][car_id].keys()
                    and "text" in results[frame_idx][car_id]["license_plate"].keys()
                ):

                    f.write(
                        "{},{},{},{},{},{},{}\n".format(
                            frame_idx,
                            car_id,
                            "[{} {} {} {}]".format(
                                results[frame_idx][car_id]["car"]["bbox"][0],
                                results[frame_idx][car_id]["car"]["bbox"][1],
                                results[frame_idx][car_id]["car"]["bbox"][2],
                                results[frame_idx][car_id]["car"]["bbox"][3],
                            ),
                            "[{} {} {} {}]".format(
                                results[frame_idx][car_id]["license_plate"]["bbox"][0],
                                results[frame_idx][car_id]["license_plate"]["bbox"][1],
                                results[frame_idx][car_id]["license_plate"]["bbox"][2],
                                results[frame_idx][car_id]["license_plate"]["bbox"][3],
                            ),
                            results[frame_idx][car_id]["license_plate"]["bbox_score"],
                            results[frame_idx][car_id]["license_plate"]["text"],
                            results[frame_idx][car_id]["license_plate"]["text_score"],
                        )
                    )
        f.close()


def license_complies_format(text):
    """
    Check if a detected license plate text matches the required format.

    Args:
        text (str): License plate text.

    Returns:
        bool: True if the license plate complies with the format, False otherwise.
    """
    if len(text) != 7:
        return False

    if (
        (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys())
        and (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys())
        and (text[2].isdigit() or text[2] in dict_char_to_int.keys())
        and (text[3].isdigit() or text[3] in dict_char_to_int.keys())
        and (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys())
        and (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys())
        and (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char.keys())
    ):
        return True
    else:
        return False


def format_license(text):
    """
    Format license plate text by replacing ambiguous characters
    using mapping dictionaries.

    Args:
        text (str): Raw OCR license plate text.

    Returns:
        str: Corrected license plate text.
    """
    formatted_plate = ""
    mapping = {
        0: dict_int_to_char,
        1: dict_int_to_char,
        4: dict_int_to_char,
        5: dict_int_to_char,
        6: dict_int_to_char,
        2: dict_char_to_int,
        3: dict_char_to_int,
    }
    for j in range(7):
        if text[j] in mapping[j].keys():
            formatted_plate += mapping[j][text[j]]
        else:
            formatted_plate += text[j]

    return formatted_plate


def read_license_plate(license_plate_crop):
    """
    Read license plate text from a cropped image using EasyOCR.

    Args:
        license_plate_crop (ndarray): Cropped license plate image.

    Returns:
        tuple: (formatted license plate text, confidence score),
               or (None, None) if not valid.
    """
    detections = reader.readtext(license_plate_crop)

    for detection in detections:
        bbox, text, score = detection
        text = text.upper().replace(" ", "")

        if license_complies_format(text):
            return format_license(text), score

    return None, None


def get_car(license_plate, vehicle_track_ids):
    """
    Match a detected license plate with its corresponding vehicle.

    Args:
        license_plate (tuple): (x1, y1, x2, y2, score, class_id).
        vehicle_track_ids (list): List of
        tracked vehicles [(x1, y1, x2, y2, car_id), ...].

    Returns:
        tuple: (x1, y1, x2, y2, car_id) if matched, else (-1, -1, -1, -1, -1).
    """
    x1, y1, x2, y2, score, class_id = license_plate

    found = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]

        # Check if license plate is inside vehicle bounding box
        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_index = j
            found = True
            break

    if found:
        return vehicle_track_ids[car_index]

    return -1, -1, -1, -1, -1
