import open_clip
import torch
from PIL import Image

# 1. Model ve preprocess fonksiyonunu yükle
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai"
)
tokenizer = open_clip.get_tokenizer("ViT-B-32")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

# 2. Sınıf metinleri listesi (araba modelleri)
class_names = [
    "a photo of an Alfa Romeo Giulia",
    "a photo of an Audi A4",
    "a photo of an Audi A6",
    "a photo of a BMW 3 Series",
    "a photo of a BMW X3",
    "a photo of a Citroen C3",
    "a photo of a Citroen C4 Grand Picasso",
    "a photo of a Dacia Logan",
    "a photo of a Dacia Spring",
    "a photo of a Fiat Bravo",
    "a photo of a Ford Fiesta",
    "a photo of a Ford Focus",
    "a photo of a Ford Fusion",
    "a photo of a Ford Mondeo",
    "a photo of a Ford Transit",
    "a photo of a Honda Civic",
    "a photo of a Hyundai i30",
    "a photo of a Kia Sportage",
    "a photo of a Maserati Levante",
    "a photo of a Mazda 2",
    "a photo of a Mini Countryman",
    "a photo of a Mitsubishi L200",
    "a photo of an Opel Astra",
    "a photo of an Opel Corsa",
    "a photo of an Opel Meriva",
    "a photo of a Peugeot 208",
    "a photo of a Peugeot 3008",
    "a photo of a Renault Captur",
    "a photo of a Seat Ibiza",
    "a photo of a Seat Leon",
    "a photo of a Skoda Fabia",
    "a photo of a Skoda Octavia",
    "a photo of a Skoda Superb",
    "a photo of a Smart Forfour",
    "a photo of a Smart Fortwo",
    "a photo of a Suzuki SX4 S-Cross",
    "a photo of a Suzuki Vitara",
    "a photo of a Tesla S",
    "a photo of a Toyota C-HR",
    "a photo of a Toyota Corolla",
    "a photo of a Toyota Yaris",
    "a photo of a Volkswagen Golf",
    "a photo of a Volkswagen Passat",
    "a photo of a Volkswagen Polo",
]

# 3. Metinleri tokenize et
text_tokens = tokenizer(class_names).to(device)

# 4. Görseli yükle ve preprocess et
image_path = "test_images/opel2.jpg"  # Tahmin yapılacak görsel
image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

# 5. Özellik çıkarımı ve benzerlik hesaplama
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text_tokens)

    # Normalize et
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # Benzerlik hesapla (cosine similarity)
    similarity = (image_features @ text_features.T).squeeze(0)
    probs = similarity.softmax(dim=0)

# 6. Sonuçları yazdır
for class_name, prob in zip(class_names, probs):
    print(f"{class_name}: {prob.item()*100:.2f}%")

print(f"\nTahmin edilen araç modeli: {class_names[probs.argmax()]}")
