# classifyv2.py

import csv
import os

import clip
import cv2
import numpy as np
import torch
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

labels = [
    "a deer",
    "a wolf",
    "a rabbit",
    "a lion",
    "a zebra",
    "a bird",
    "a tiger",
    "a bear",
    "an owl",
    "an elephant",
]
text_tokens = clip.tokenize(labels).to(device)


def get_masks(image, mask_generator):
    masks = mask_generator.generate(image)
    return masks


def crop_mask_region(image, mask):
    m = mask["segmentation"].astype(np.uint8) * 255
    x, y, w, h = cv2.boundingRect(m)
    cropped = image[y : y + h, x : x + w]
    return cropped, (x, y, w, h)


def predict_with_clip(image_crop):
    pil_image = Image.fromarray(cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB))
    image_input = preprocess(pil_image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_tokens)
        logits_per_image = image_features @ text_features.T
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]
    top_label = labels[np.argmax(probs)]
    confidence = np.max(probs)
    return top_label, confidence


def process_image(image_path, mask_generator, output_path, csv_writer, segment_dir):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = get_masks(image_rgb, mask_generator)
    image_name = os.path.basename(image_path)

    count = 0
    for i, mask in enumerate(masks):
        area = mask["area"]
        if area < 5000 or mask["stability_score"] < 0.9:
            continue

        crop, (x, y, w, h) = crop_mask_region(image, mask)
        label, conf = predict_with_clip(crop)

        # Görsele yaz
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            image,
            f"{label} ({conf:.2f})",
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        # CSV log satırı yaz
        csv_writer.writerow([image_name, label, round(conf, 2), x, y, w, h])

        # Segment görselini kaydet
        segment_path = os.path.join(segment_dir, f"{image_name}_seg{i}.png")
        cv2.imwrite(segment_path, crop)

        count += 1

    cv2.imwrite(output_path, image)
    print(f"{image_name} işlendi - {count} segment")


def main():
    checkpoint_path = "models/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    input_folder = "data"
    output_folder = "outputs_classifiedv2"
    segment_dir = "segments"
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(segment_dir, exist_ok=True)

    csv_path = "log.csv"
    with open(csv_path, mode="w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["image", "label", "confidence", "x", "y", "w", "h"])

        for filename in sorted(os.listdir(input_folder)):
            if filename.endswith(".jpg"):
                image_path = os.path.join(input_folder, filename)
                output_path = os.path.join(output_folder, f"classified_{filename}")
                process_image(
                    image_path, mask_generator, output_path, csv_writer, segment_dir
                )

    print("✅ Tüm görseller işlendi ve log.csv oluşturuldu.")


if __name__ == "__main__":
    main()
