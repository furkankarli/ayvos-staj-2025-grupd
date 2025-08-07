# classify.py

import os

import clip  # OpenAI'nin CLIP kütüphanesi
import cv2
import numpy as np
import torch
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

# 🧠 1. CLIP modeli yükle
device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# 📋 2. Etiket metinleri
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


# 📤 3. Segmentasyon maskelerini oluştur (aynı segment.py içindeki gibi)
def get_masks(image, mask_generator):
    masks = mask_generator.generate(image)
    return masks


# ✂️ 4. Maskeyi kullanarak hayvanı görüntüden kırp
def crop_mask_region(image, mask):
    m = mask["segmentation"].astype(np.uint8) * 255
    x, y, w, h = cv2.boundingRect(m)
    cropped = image[y : y + h, x : x + w]
    return cropped, (x, y, w, h)


# 🔍 5. CLIP ile tahmin et
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


# 🖼️ 6. Görseli işle ve üzerine yaz
def process_image(image_path, mask_generator, output_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = get_masks(image_rgb, mask_generator)

    for mask in masks:
        area = mask["area"]
        if area < 5000:  # çok küçük maskeleri at
            continue

        crop, (x, y, w, h) = crop_mask_region(image, mask)
        label, conf = predict_with_clip(crop)

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

    cv2.imwrite(output_path, image)
    print(f"Kaydedildi: {output_path}")


# 🚀 7. Ana fonksiyon
def main():
    checkpoint_path = "models/sam_vit_h_4b8939.pth"
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device)
    mask_generator = SamAutomaticMaskGenerator(sam)

    input_folder = "data"
    output_folder = "outputs_classified"
    os.makedirs(output_folder, exist_ok=True)

    for filename in sorted(os.listdir(input_folder)):
        if filename.endswith(".jpg"):
            image_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"classified_{filename}")
            process_image(image_path, mask_generator, output_path)


if __name__ == "__main__":
    main()
