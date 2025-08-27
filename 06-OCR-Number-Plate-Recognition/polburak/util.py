import string

import easyocr

reader = easyocr.Reader(["en"], gpu=False)

dict_char_to_int = {"O": "0", "I": "1", "J": "3", "A": "4", "G": "6", "S": "5"}
dict_int_to_char = {"0": "O", "1": "I", "3": "J", "4": "A", "6": "G", "5": "S"}


def license_complies_format(text):
    if len(text) != 7:
        return False
    if (
        (text[0] in string.ascii_uppercase or text[0] in dict_int_to_char.keys())
        and (text[1] in string.ascii_uppercase or text[1] in dict_int_to_char.keys())
        and (text[2] in "0123456789" or text[2] in dict_char_to_int.keys())
        and (text[3] in "0123456789" or text[3] in dict_char_to_int.keys())
        and (text[4] in string.ascii_uppercase or text[4] in dict_int_to_char.keys())
        and (text[5] in string.ascii_uppercase or text[5] in dict_int_to_char.keys())
        and (text[6] in string.ascii_uppercase or text[6] in dict_int_to_char.keys())
    ):
        return True
    return False


def format_license(text):
    license_plate_ = ""
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
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]
    return license_plate_


def read_license_plate(license_plate_crop):
    detections = reader.readtext(license_plate_crop)
    for detection in detections:
        bbox, text, score = detection
        text = text.upper().replace(" ", "")
        if license_complies_format(text):
            return format_license(text), score
    return None, None


def get_car(license_plate, vehicle_track_ids):
    x1, y1, x2, y2, score, class_id = license_plate
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]
        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            return xcar1, ycar1, xcar2, ycar2, car_id
    return -1, -1, -1, -1, -1
