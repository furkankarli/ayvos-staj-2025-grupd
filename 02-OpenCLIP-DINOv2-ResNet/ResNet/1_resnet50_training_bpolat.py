import json
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
            print(f"Image load error: {self.images[idx]}, {e}")
            return torch.zeros(3, 224, 224), 0

        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label


transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_dataset = ImageDataset("train.json", transform)
val_dataset = ImageDataset("val.json", transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

num_classes = len(set(train_dataset.labels))

model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

num_epochs = 10

start_time = time.time()

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
        f"[Epoch {epoch+1}/{num_epochs}] Loss: {running_loss:.4f} "
        f"- Train Acc: {train_acc:.2f}%"
    )

end_time = time.time()
print(f"\nEğitim süresi: {(end_time - start_time):.2f} saniye")

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

# Modeli kaydet (.pth)
torch.save(model.state_dict(), "resnet50_trained.pth")
print("Model 'resnet50_trained.pth' olarak kaydedildi.")

# Label encoder sınıf isimlerini JSON olarak kaydet
classes = train_dataset.le.classes_.tolist()
with open("label_encoder.json", "w") as f:
    json.dump(classes, f)
print("Label encoder sınıf isimleri 'label_encoder.json' olarak kaydedildi.")
