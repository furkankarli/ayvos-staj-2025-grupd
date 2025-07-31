import glob
import os

import torch
from open_clip import create_model_and_transforms, tokenize
from PIL import Image, ImageDraw, ImageFont

MODEL_NAME = "ViT-B-32-quickgelu"
CHECKPOINT_PATH = "logs/arac-modeli-zengin-veri-v1/checkpoints/epoch_39.pt"
TEST_IMAGES_FOLDER = "test_images"
RESULTS_FOLDER = "results"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SIMILARITY_THRESHOLD = 0.90

PROMPT_TEMPLATES = [
    "Bir fotoğraf, model: {}.",
    "Bu bir {} arabası.",
    "Yüksek çözünürlüklü bir {} resmi.",
    "Stüdyo çekimi bir {}.",
    "{} aracının bir fotoğrafı.",
]

CLASS_NAMES_CLEAN = {
    "honda-civic": "Honda Civic",
    "mitsubishi-l200": "Mitsubishi L200",
    "opel-astra": "Opel Astra",
    "bmw-3-series": "BMW 3 Serisi",
    "toyota-c-hr": "Toyota C-HR",
}

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_SIZE_HEADER = 28
FONT_SIZE_BODY = 22
IMG_WIDTH = 400
PADDING = 20
model, _, preprocess = create_model_and_transforms(
    MODEL_NAME, pretrained=CHECKPOINT_PATH, device=DEVICE
)
model.eval()
print(f"✅ Model ve checkpoint başarıyla yüklendi: {CHECKPOINT_PATH}")

os.makedirs(RESULTS_FOLDER, exist_ok=True)

try:
    font_header = ImageFont.truetype(FONT_PATH, FONT_SIZE_HEADER)
    font_body = ImageFont.truetype(FONT_PATH, FONT_SIZE_BODY)
except IOError:
    print(f"❌ Font dosyası bulunamadı: {FONT_PATH}. Varsayılan font kullanılacak.")
    font_header = ImageFont.load_default()
    font_body = ImageFont.load_default()

all_texts = []
for class_name in CLASS_NAMES_CLEAN.values():
    prompts = [template.format(class_name) for template in PROMPT_TEMPLATES]
    all_texts.extend(prompts)
tokenized_texts = tokenize(all_texts).to(DEVICE)
image_files = (
    glob.glob(os.path.join(TEST_IMAGES_FOLDER, "*.jpg"))
    + glob.glob(os.path.join(TEST_IMAGES_FOLDER, "*.jpeg"))
    + glob.glob(os.path.join(TEST_IMAGES_FOLDER, "*.png"))
)

if not image_files:
    print(f"❌ '{TEST_IMAGES_FOLDER}' klasöründe hiç görsel bulunamadı.")
    exit()

print(f"\nFound {len(image_files)} images to test in '{TEST_IMAGES_FOLDER}' folder.")
print("-" * 40)

for image_path in image_files:
    try:
        original_image = Image.open(image_path).convert("RGB")
        image_for_model = preprocess(original_image).unsqueeze(0).to(DEVICE)

        with torch.no_grad(), torch.amp.autocast(device_type=DEVICE):
            image_features = model.encode_image(image_for_model)
            text_features = model.encode_text(tokenized_texts)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            num_classes = len(CLASS_NAMES_CLEAN)
            num_templates = len(PROMPT_TEMPLATES)
            text_features = text_features.reshape(num_classes, num_templates, -1).mean(
                dim=1
            )
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

        similarity_scores = similarity.squeeze().cpu().tolist()
        class_names_list = list(CLASS_NAMES_CLEAN.values())
        max_score = max(similarity_scores)
        predicted_class = class_names_list[similarity_scores.index(max_score)]

        aspect_ratio = original_image.height / original_image.width
        img_height = int(IMG_WIDTH * aspect_ratio)
        resized_image = original_image.resize((IMG_WIDTH, img_height))

        total_width = IMG_WIDTH + 400
        total_height = max(img_height, 400)
        result_image = Image.new("RGB", (total_width, total_height), "white")
        result_image.paste(resized_image, (0, 0))
        draw = ImageDraw.Draw(result_image)

        text_x = IMG_WIDTH + PADDING
        text_y = PADDING

        is_known = max_score >= SIMILARITY_THRESHOLD
        main_prediction_text = (
            f"Tahmin: {predicted_class}" if is_known else "Tahmin: Bilinmeyen Araç"
        )
        main_color = "green" if is_known else "orange"
        draw.text(
            (text_x, text_y), main_prediction_text, font=font_header, fill=main_color
        )
        text_y += FONT_SIZE_HEADER + 5

        score_text = f"Skor: %{max_score*100:.2f}"
        draw.text((text_x, text_y), score_text, font=font_body, fill="black")
        text_y += FONT_SIZE_BODY + PADDING

        draw.text((text_x, text_y), "--- Tüm Skorlar ---", font=font_body, fill="gray")
        text_y += FONT_SIZE_BODY + 10

        sorted_results = sorted(
            zip(class_names_list, similarity_scores), key=lambda x: x[1], reverse=True
        )
        for class_name, score in sorted_results:
            line = f"• {class_name:<20} : %{score*100:.2f}"
            draw.text((text_x, text_y), line, font=font_body, fill="black")
            text_y += FONT_SIZE_BODY + 5

        base_filename = os.path.basename(image_path)
        name, ext = os.path.splitext(base_filename)
        output_path = os.path.join(RESULTS_FOLDER, f"{name}_result.png")
        result_image.save(output_path)
        print(f"Sonuç görseli kaydedildi: {output_path}")

    except Exception as e:
        print(f"\n'{os.path.basename(image_path)}' işlenirken bir hata oluştu: {e}")

print("-" * 40)
print("Test ve görselleştirme işlemi tamamlandı.")
