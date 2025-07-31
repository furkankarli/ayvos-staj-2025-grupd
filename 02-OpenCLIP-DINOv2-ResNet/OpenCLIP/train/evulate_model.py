import csv
import os

import matplotlib.pyplot as plt
import seaborn as sns
import torch
from open_clip import create_model_and_transforms, tokenize
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm

MODEL_NAME = "ViT-B-32-quickgelu"
CHECKPOINT_PATH = "logs/arac-modeli-zengin-veri-v1/checkpoints/epoch_39.pt"
EVAL_CSV_PATH = "test.csv"
RESULTS_PLOT_PATH = "final_confusion_matrix.png"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLASS_DEFINITIONS = {
    "honda-civic": {"model_name": "Honda Civic", "car_type": "sedan"},
    "mitsubishi-l200": {"model_name": "Mitsubishi L200", "car_type": "kamyonet"},
    "opel-astra": {"model_name": "Opel Astra", "car_type": "hatchback"},
    "bmw-3-series": {"model_name": "BMW 3 Serisi", "car_type": "sedan"},
    "toyota-c-hr": {"model_name": "Toyota C-HR", "car_type": "SUV"},
}

PROMPT_TEMPLATES = [
    "Bir {car_type} olan {model_name} aracının yüksek çözünürlüklü bir fotoğrafı.",
    "Bu bir {model_name}, bir {car_type}.",
    "{model_name} modelinin detaylı bir stüdyo çekimi.",
    "Dış mekanda çekilmiş bir {model_name} ({car_type}).",
    "İşte bir {model_name} aracının net bir görseli.",
]

print("Model ve metinler hazırlanıyor...")
model, _, preprocess = create_model_and_transforms(
    MODEL_NAME, pretrained=CHECKPOINT_PATH, device=DEVICE
)
model.eval()

all_texts = []
class_names_list = []
for class_key, definitions in CLASS_DEFINITIONS.items():
    class_names_list.append(definitions["model_name"])
    prompts = [
        template.format(
            model_name=definitions["model_name"], car_type=definitions["car_type"]
        )
        for template in PROMPT_TEMPLATES
    ]
    all_texts.extend(prompts)
tokenized_texts = tokenize(all_texts).to(DEVICE)
print("✅ Model ve metinler hazır.")

true_labels = []
predicted_labels = []

print("\n'{}' dosyasındaki görseller test ediliyor...".format(EVAL_CSV_PATH))
with open(EVAL_CSV_PATH, "r") as csvfile:
    reader = csv.reader(csvfile)
    header = next(reader)

    rows = list(reader)
    for row in tqdm(rows, desc="Değerlendirme"):
        image_path, true_class_key = row

        true_class_name = CLASS_DEFINITIONS.get(true_class_key, {}).get("model_name")

        if not true_class_name:
            print(
                "Uyarı: '{}' anahtarı CLASS_DEFINITIONS içinde bulunamadı. "
                "Satır atlanıyor.".format(true_class_key)
            )
            continue

        true_labels.append(true_class_name)

        try:
            image = (
                preprocess(Image.open(image_path).convert("RGB"))
                .unsqueeze(0)
                .to(DEVICE)
            )
            with torch.no_grad(), torch.amp.autocast(device_type=DEVICE):
                image_features = model.encode_image(image)
                text_features = model.encode_text(tokenized_texts)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                num_classes = len(CLASS_DEFINITIONS)
                num_templates = len(PROMPT_TEMPLATES)
                text_features = text_features.reshape(
                    num_classes, num_templates, -1
                ).mean(dim=1)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

            pred_index = similarity.argmax().item()
            predicted_class_name = class_names_list[pred_index]
            predicted_labels.append(predicted_class_name)

            is_correct = (
                "DOĞRU" if true_class_name == predicted_class_name else "❌ YANLIŞ"
            )
            tqdm.write(
                "Görsel: {:<15} | Gerçek: {:<15} | "
                "Tahmin: {:<15} | Sonuç: {}".format(
                    os.path.basename(image_path),
                    true_class_name,
                    predicted_class_name,
                    is_correct,
                )
            )

        except Exception as e:
            tqdm.write("Hata: {} işlenemedi. Hata: {}".format(image_path, e))
            true_labels.pop()

if not true_labels:
    print(
        "Hiç geçerli veri değerlendirilemedi. "
        "CSV dosyasını ve CLASS_DEFINITIONS'ı kontrol edin."
    )
    exit()

accuracy = accuracy_score(true_labels, predicted_labels)
print("\n" + "=" * 50)
print("NİHAİ DEĞERLENDİRME SONUCU")
print(f"   Toplam Değerlendirilen Görsel: {len(true_labels)}")
print(f"   Genel Doğruluk: %{accuracy * 100:.2f}")
print("=" * 50)

cm = confusion_matrix(true_labels, predicted_labels, labels=class_names_list)

plt.figure(figsize=(12, 10))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names_list,
    yticklabels=class_names_list,
)
plt.title(f"Karışıklık Matrisi (Genel Doğruluk: %{accuracy * 100:.2f})", fontsize=16)
plt.ylabel("Gerçek Sınıf (True Label)", fontsize=12)
plt.xlabel("Tahmin Edilen Sınıf (Predicted Label)", fontsize=12)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()

plt.savefig(RESULTS_PLOT_PATH)
print("Sonuç grafiği başarıyla kaydedildi: {}".format(RESULTS_PLOT_PATH))
