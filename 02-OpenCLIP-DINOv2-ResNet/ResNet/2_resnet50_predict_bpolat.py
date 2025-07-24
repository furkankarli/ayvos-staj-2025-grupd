import json
import os

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Görüntü dönüşümü (train ile aynı)
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def load_label_encoder(json_path):
    with open(json_path, "r") as f:
        class_names = json.load(f)
    return class_names


def load_model(model_path, num_classes):
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict_image(model, class_names, image_path):
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        _, pred = torch.max(outputs, 1)
    class_idx = pred.item()
    class_name = class_names[class_idx]

    return class_name


def main():
    model_path = "resnet50_trained.pth"
    encoder_path = "label_encoder.json"
    input_folder = "input"  # Tahmin yapmak istediğin görsellerin klasörü

    class_names = load_label_encoder(encoder_path)
    model = load_model(model_path, num_classes=len(class_names))

    for filename in os.listdir(input_folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image_path = os.path.join(input_folder, filename)
            prediction = predict_image(model, class_names, image_path)
            print(f"{filename} → Tahmin: {prediction}")


if __name__ == "__main__":
    main()
