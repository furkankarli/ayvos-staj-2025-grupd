import os

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

# 1. Teknik Ayarlar
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    DEVICE = torch.device("mps")
    print("MPS kullanılacak (Apple Silicon GPU)")
else:
    DEVICE = torch.device("cpu")
    print("CPU kullanılacak")

model_ckpt = (
    "facebook/dinov2-base"  # Eğitirken bu ile başlamışsan config olarak aynısını kullan
)
num_classes = 44

# Senin class_names dizin:
class_names = [
    "alfa-romeo-giulia",
    "audi-a4",
    "audi-a6",
    "bmw-3-series",
    "bmw-x3",
    "citroen-c3",
    "citroen-c4-grand-picasso",
    "dacia-logan",
    "dacia-spring",
    "fiat-bravo",
    "ford-fiesta",
    "ford-focus",
    "ford-fusion",
    "ford-mondeo",
    "ford-transit",
    "honda-civic",
    "hyundai-i30",
    "kia-sportage",
    "maserati-levante",
    "mazda-2",
    "mini-countryman",
    "mitsubishi-l200",
    "opel-astra",
    "opel-corsa",
    "opel-meriva",
    "peugeot-208",
    "peugeot-3008",
    "renault-captur",
    "seat-ibiza",
    "seat-leon",
    "skoda-fabia",
    "skoda-octavia",
    "skoda-superb",
    "smart-forfour",
    "smart-fortwo",
    "suzuki-sx4-s-cross",
    "suzuki-vitara",
    "tesla-s",
    "toyota-c-hr",
    "toyota-corolla",
    "toyota-yaris",
    "volkswagen-golf",
    "volkswagen-passat",
    "volkswagen-polo",
]

# 2. Model ve Processor Yükle
model = AutoModelForImageClassification.from_pretrained(
    model_ckpt, num_labels=num_classes, ignore_mismatched_sizes=True
)
model.load_state_dict(
    torch.load("dinov2_finetuned.pt", map_location=DEVICE)
)  # .pth dosyası adı burada!
model = model.to(DEVICE)
model.eval()

processor = AutoImageProcessor.from_pretrained(model_ckpt)

# 3. Test klasöründe resim dosyalarını bul
test_dir = "test_imagesv0"
image_files = [
    f for f in os.listdir(test_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

# 4. Her fotoğrafı tahmin et ve sonucu yazdır
results = []
with torch.no_grad():
    for img_name in image_files:
        img_path = os.path.join(test_dir, img_name)
        pil_img = Image.open(img_path).convert("RGB")
        inputs = processor(images=pil_img, return_tensors="pt").to(DEVICE)
        outputs = model(**inputs)
        pred_id = outputs.logits.argmax(dim=1).item()
        pred_class = class_names[pred_id]
        print(f"{img_name} --> Tahmin: {pred_class}")
        results.append({"file": img_name, "prediction": pred_class})

# 5. Opsiyonel: Sonuçları CSV'ye yaz
import csv

with open("test_predictions.csv", "w") as f:
    writer = csv.DictWriter(f, fieldnames=["file", "prediction"])
    writer.writeheader()
    writer.writerows(results)
