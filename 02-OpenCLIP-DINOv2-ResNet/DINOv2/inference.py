import os

import numpy as np
import torch
from PIL import Image

# ---- SINIF İSİMLERİNİ YÜKLE (Klasör sıralı mapping!) ----
from torchvision import datasets, transforms
from transformers import Dinov2ForImageClassification

# ---- YOL AYARLARI ----
base_path = "dino/Dinov2"
test_images_dir = os.path.join(f"{base_path}/test_images")
model_ckpt_path = os.path.join(f"{base_path}/dinov2_finetuned_car5.pth")
labels_json_path = os.path.join(f"{base_path}/labels.json")


_dummy = datasets.ImageFolder(os.path.join(base_path, "split_dataset", "train"))
class_names = _dummy.classes

# Alternatif: labels.json sırası da ekle ama asla mappingi bozma!
# with open(labels_json_path, "r") as f:
#     label_map = json.load(f)
# idx2label = [label_map[k] for k in sorted(label_map.keys())]

# ---- MODELİ YÜKLE ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Dinov2ForImageClassification.from_pretrained(
    "facebook/dinov2-base", num_labels=len(class_names), ignore_mismatched_sizes=True
).to(device)
model.load_state_dict(torch.load(model_ckpt_path, map_location=device))
model.eval()

print(f"\n✅ Model ve checkpoint başarıyla yüklendi: {model_ckpt_path}\n")

# ---- TRANSFORM ----
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

image_paths = [
    os.path.join(test_images_dir, fname)
    for fname in os.listdir(test_images_dir)
    if fname.lower().endswith((".jpg", ".jpeg", ".png"))
]
image_paths.sort()
print(
    f"Found {len(image_paths)} images to test in "
    f"'{os.path.basename(test_images_dir)}' folder."
)
print("-" * 40)

min_conf = 0.90  # %90 altında "bilinmeyen" verilecek


def pretty_scores(probabilities, class_names):
    sorted_idx = np.argsort(probabilities)[::-1]
    rows = []
    for idx in sorted_idx:
        rows.append(f"    🔹 {class_names[idx]:18} : %{probabilities[idx]*100:.2f}")
    return "\n" + "\n".join(rows)


for img_path in image_paths:
    img_name = os.path.basename(img_path)
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x).logits.squeeze()
        probs = torch.softmax(logits, dim=0).cpu().numpy()
        top_idx = np.argmax(probs)
        pred_score = probs[top_idx]
        pred_label = class_names[top_idx]

    print(f"\n📸 Test Sonuçları: {img_name}")

    if pred_score >= min_conf:
        print(f"  ✅ Tahmin: {pred_label} (Skor: %{pred_score*100:.2f})")
    else:
        print(
            f"  ⚠️  Tahmin: Bilinmeyen Araç (En Yakın Tahmin: {pred_label}, "
            f"Skor: %{pred_score*100:.2f})"
        )
        print("     (Sebep: En yüksek skor, %90 olan eşik değerinin altındadır)")

    print("  -- Tüm Skorlar --")
    print(pretty_scores(probs, class_names))

print("-" * 40)
print("✅ Test işlemi tamamlandı.")
