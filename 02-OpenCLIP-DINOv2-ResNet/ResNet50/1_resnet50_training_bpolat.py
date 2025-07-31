import json
import math
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Veri bölme ve JSON oluşturma ---
def generate_split_json(data_dir="dataset", label_json="data.json", output_dir="."):
    with open(label_json, "r") as f:
        label_map = json.load(f)

    train_list, val_list, test_list = [], [], []

    for folder_name, model_name in label_map.items():
        folder_path = os.path.join(data_dir, folder_name)
        if not os.path.isdir(folder_path):
            print(f"Klasör yok: {folder_path}")
            continue

        all_images = sorted(
            [img for img in os.listdir(folder_path) if img.lower().endswith(".jpg")]
        )

        total = len(all_images)
        if total == 0:
            print(f"Görsel yok: {folder_path}")
            continue

        train_end = math.floor(0.6 * total)
        val_end = train_end + math.floor(0.2 * total)

        for i, img_name in enumerate(all_images):
            item = {
                "image_path": os.path.join(data_dir, folder_name, img_name).replace(
                    "\\", "/"
                ),
                "model_name": model_name,
            }

            if i < train_end:
                train_list.append(item)
            elif i < val_end:
                val_list.append(item)
            else:
                test_list.append(item)

    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "train.json"), "w") as f:
        json.dump(train_list, f, indent=2)
    with open(os.path.join(output_dir, "val.json"), "w") as f:
        json.dump(val_list, f, indent=2)
    with open(os.path.join(output_dir, "test.json"), "w") as f:
        json.dump(test_list, f, indent=2)

    print(f" train.json: {len(train_list)} örnek")
    print(f" val.json: {len(val_list)} örnek")
    print(f" test.json: {len(test_list)} örnek")


# --- Dataset ---
class ImageDataset(Dataset):
    def __init__(self, json_path, transform=None):
        with open(json_path, "r") as f:
            self.data = json.load(f)
        self.transform = transform
        self.images = []
        self.labels = []
        for item in self.data:
            img_path = item["image_path"]
            if not os.path.isabs(img_path):
                img_path = os.path.join(os.path.dirname(json_path), img_path)
            self.images.append(img_path)
            self.labels.append(item["model_name"])

        self.le = LabelEncoder()
        self.labels = self.le.fit_transform(self.labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            image = Image.open(self.images[idx]).convert("RGB")
        except Exception as e:
            print(f"Hata - Görüntü açılamadı: {self.images[idx]}, {e}")
            # Boş görüntü döndür veya bir sonraki örneğe geç
            return self.__getitem__((idx + 1) % len(self.images))

        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label


# --- Ana program ---
def main():
    data_dir = "dataset"
    label_json = "data.json"
    output_dir = "."

    # 1) Veri böl ve JSON oluştur
    generate_split_json(data_dir, label_json, output_dir)

    # 2) Transform tanımı
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # 3) Dataset ve DataLoader
    train_dataset = ImageDataset(os.path.join(output_dir, "train.json"), transform)
    val_dataset = ImageDataset(os.path.join(output_dir, "val.json"), transform)
    test_dataset = ImageDataset(os.path.join(output_dir, "test.json"), transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    test_loader = DataLoader(test_dataset, batch_size=16)

    # 4) Model oluştur
    num_classes = len(set(train_dataset.labels))
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    num_epochs = 20
    start_time = time.time()

    # 5) Eğitim döngüsü
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        print(
            f"[Epoch {epoch+1}/{num_epochs}] "
            f"Loss: {running_loss:.4f} "
            f"Train Acc: {train_acc:.2f}%"
        )

    end_time = time.time()
    print(f"\n Eğitim süresi: {(end_time - start_time):.2f} saniye")

    # 6) Validation doğruluğu
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = 100 * correct / total
    print(f"Validation Accuracy: {val_acc:.2f}%")

    # 7) Test doğruluğu
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = 100 * correct / total
    print(f"Test Accuracy: {test_acc:.2f}%")

    # 8) Modeli kaydet
    torch.save(model.state_dict(), "resnet50_trained.pth")
    print("Model 'resnet50_trained.pth' olarak kaydedildi.")

    # 9) Label encoder sınıf isimlerini kaydet
    classes = train_dataset.le.classes_.tolist()
    with open("label_encoder.json", "w") as f:
        json.dump(classes, f)
    print("Sınıf isimleri 'label_encoder.json' olarak kaydedildi.")


if __name__ == "__main__":
    main()
