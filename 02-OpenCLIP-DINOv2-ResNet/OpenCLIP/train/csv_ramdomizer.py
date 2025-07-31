import csv
import os
import random


def generate_new_captions(input_csv_path, output_csv_path):
    model_map = {
        "honda-civic": "Honda Civic sedan",
        "mitsubishi-l200": "Mitsubishi L200 kamyonet",
        "opel-astra": "Opel Astra hatchback",
        "bmw-3-series": "BMW 3 serisi sedan",
        "toyota-c-hr": "Toyota C-HR SUV",
    }

    templates = [
        "Bir {model_name} fotoğrafı, {angle_desc} açısından.",
        "{model_name} aracının {angle_desc} görünümü.",
        "Bu bir {model_name}, {angle_desc} perspektifinden.",
        "Yüksek çözünürlüklü bir {model_name} resmi, {angle_desc} açısından çekilmiş.",
        "Stüdyo çekimi bir {model_name}, {angle_desc}.",
    ]

    with open(input_csv_path, "r") as infile, open(
        output_csv_path, "w", newline=""
    ) as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        header = next(reader)
        writer.writerow(header)

        for row in reader:
            image_path, model_key = row
            try:
                img_number = int(os.path.basename(image_path).split(".")[0])
                angle = img_number % 360

                angle_desc = "bilinmeyen açı"
                if 0 <= angle < 20 or 340 <= angle < 360:
                    angle_desc = "ön"
                elif 20 <= angle < 70:
                    angle_desc = "ön-yan"
                elif 70 <= angle < 110:
                    angle_desc = "yan"
                elif 110 <= angle < 160:
                    angle_desc = "arka-yan"
                elif 160 <= angle < 200:
                    angle_desc = "arka"
                elif 200 <= angle < 250:
                    angle_desc = "arka-yan"
                elif 250 <= angle < 290:
                    angle_desc = "yan"
                elif 290 <= angle < 340:
                    angle_desc = "ön-yan"

            except (ValueError, IndexError):
                angle_desc = "bir açı"

            chosen_template = random.choice(templates)
            full_model_name = model_map.get(model_key, model_key.replace("-", " "))
            new_caption = chosen_template.format(
                model_name=full_model_name, angle_desc=angle_desc
            )

            writer.writerow([image_path, new_caption])


generate_new_captions("train.csv", "train_new.csv")
generate_new_captions("val.csv", "val_new.csv")
generate_new_captions("test.csv", "test_new.csv")
