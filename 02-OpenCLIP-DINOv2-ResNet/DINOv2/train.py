import os

import torch
from dataset import get_data_loaders
from model import get_model
from tqdm import tqdm

base_path = "dino/Dinov2"
train_dir = os.path.join(f"{base_path}/split_dataset/train")
val_dir = os.path.join(f"{base_path}/split_dataset/val")
test_dir = os.path.join(f"{base_path}/split_dataset/test")

batch_size = 32
num_epochs = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan cihaz: {device}")

train_loader, val_loader, test_loader, class_names = get_data_loaders(
    train_dir, val_dir, test_dir, batch_size=batch_size
)
num_classes = len(class_names)

model = get_model(num_classes, device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for images, labels in tqdm(loader, desc="Eğitim"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images).logits
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(loader)
    return avg_loss


def evaluate(model, loader, loss_fn, device):
    model.eval()
    correct, total, total_loss = 0, 0, 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Doğrulama/Test"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).logits
            loss = loss_fn(outputs, labels)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item()
    accuracy = correct / total
    avg_loss = total_loss / len(loader)
    return avg_loss, accuracy


for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}:")
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    print(f"  Train Loss: {train_loss:.4f}")
    print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

test_loss, test_acc = evaluate(model, test_loader, criterion, device)
print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")

# Modeli kaydet
save_path = os.path.join(base_path, "dinov2_finetuned_car5.pth")
torch.save(model.state_dict(), save_path)
print("Model kaydedildi:", save_path)
