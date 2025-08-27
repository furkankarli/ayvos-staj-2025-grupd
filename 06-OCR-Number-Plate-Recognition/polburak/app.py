import os
import re

import cv2
import easyocr
from db import fetch_plates, insert_plate
from flask import Flask, render_template, request
from ultralytics import YOLO

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"

# Plaka dedektörü
lp_model = YOLO("./models/license_plate_detector.pt")

# OCR reader
reader = easyocr.Reader(["tr", "en"], gpu=False)


def read_license_plate_direct(license_plate_crop):
    detections = reader.readtext(license_plate_crop)
    if not detections:
        return None

    full_text = ""
    for bbox, text, score in detections:
        text = text.replace(" ", "").upper()
        full_text += text
    return full_text


def extract_tr_plate(text):
    text = text.replace(" ", "").upper()
    if text.startswith("TR"):
        text = text[2:]
    match = re.search(r"\d{2}[A-Z]{1,2}\d{2,4}", text)
    if match:
        return match.group()
    return text


@app.route("/", methods=["GET", "POST"])
def index():
    result_texts = []
    image_path = None

    if request.method == "POST":
        file = request.files["file"]
        if file:
            filename = file.filename
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(save_path)

            frame = cv2.imread(save_path)

            plates = lp_model(frame)[0]
            for idx, plate in enumerate(plates.boxes.data.tolist()):
                x1, y1, x2, y2, score, cls = map(int, plate[:6])

                margin_x = int((x2 - x1) * 0.3)
                margin_y = int((y2 - y1) * 0.3)
                crop = frame[
                    max(0, y1 - margin_y) : y2 + margin_y,
                    max(0, x1 - margin_x) : x2 + margin_x,
                ]

                if crop.size == 0:
                    result_texts.append(f"Plate {idx+1}: OCR failed (crop too small)")
                    continue

                text = read_license_plate_direct(crop)
                if text:
                    plate_text = extract_tr_plate(text)
                    result_texts.append(f"Plate {idx+1}: {plate_text}")

                    # Veritabanına kaydet
                    insert_plate(plate_text, filename)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(
                        frame,
                        plate_text,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )
                else:
                    result_texts.append(f"Plate {idx+1}: OCR failed")
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            output_path = os.path.join(
                app.config["UPLOAD_FOLDER"], "result_" + filename
            )
            cv2.imwrite(output_path, frame)
            image_path = output_path

    plates_history = fetch_plates()
    return render_template(
        "index.html",
        results=result_texts,
        image_path=image_path,
        history=plates_history,
    )


if __name__ == "__main__":
    app.run(debug=True)
