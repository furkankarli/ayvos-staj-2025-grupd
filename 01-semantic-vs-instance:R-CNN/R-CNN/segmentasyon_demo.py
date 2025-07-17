from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image

# 📥 1. İnternetten örnek görsel çek (COCO test görseli)
img_url = "http://images.cocodataset.org/val2017/000000039769.jpg"
response = requests.get(img_url)
image = Image.open(BytesIO(response.content)).convert("RGB")

# 🎨 2. Görseli Tensöre çevir
transform = T.Compose([T.ToTensor()])
img_tensor = transform(image)

# 📦 3. Mask R-CNN Modelini yükle (pretrained)
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

# ⚙️ 4. Inference (tahmin)
with torch.no_grad():
    predictions = model([img_tensor])[0]

# 🎯 5. Instance Çıktılarını çiz
img_np = np.array(image)
instance_img = img_np.copy()

for i in range(len(predictions["scores"])):
    score = predictions["scores"][i].item()
    if score < 0.8:
        continue
    mask = predictions["masks"][i, 0].mul(255).byte().cpu().numpy()
    label = predictions["labels"][i].item()
    color = np.random.randint(0, 255, (3,), dtype=int)

    instance_img[mask > 128] = instance_img[mask > 128] * 0.5 + color * 0.5

# 🟩 6. Semantic Maskeyi (her piksele bir sınıf gibi) oluştur
semantic_mask = np.zeros_like(img_np[:, :, 0])
for i in range(len(predictions["scores"])):
    score = predictions["scores"][i].item()
    if score < 0.8:
        continue
    mask = predictions["masks"][i, 0].cpu().numpy()
    label = predictions["labels"][i].item()
    semantic_mask[mask > 0.5] = label  # sınıf ID'sini yaz

# 🎥 7. Görselleştirme
plt.figure(figsize=(16, 6))

plt.subplot(1, 3, 1)
plt.title("Orijinal Görüntü")
plt.imshow(image)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title("Instance Segmentation")
plt.imshow(instance_img)
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title("Semantic Segmentation (mask gibi)")
plt.imshow(semantic_mask, cmap="tab20")
plt.axis("off")

plt.tight_layout()
plt.show()
