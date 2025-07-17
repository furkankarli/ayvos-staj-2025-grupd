# YOLOv8-Seg ve Mask R-CNN ile Semantic ve Instance Segmentation

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn
from ultralytics import YOLO

image_path = "Segmentasyon/photo.jpg"  # Deneme görseli
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Görseli yeniden boyutlandır (model uyumluluğu için)
image_resized = cv2.resize(image_rgb, (640, 640))

# === YOLOv8-Seg ile Instance Segmentation ===
yolo_model = YOLO("yolov8n-seg.pt")
yolo_result = yolo_model(image_resized)[0]

# YOLO'dan maskeleri al
instance_masks_yolo = (
    yolo_result.masks.data.cpu().numpy() if yolo_result.masks is not None else []
)

# === Mask R-CNN ile Instance Segmentation ===
maskrcnn = maskrcnn_resnet50_fpn(pretrained=True)
maskrcnn.eval()

transform = T.Compose([T.ToPILImage(), T.ToTensor()])

with torch.no_grad():
    input_tensor = transform(image_resized)
    prediction = maskrcnn([input_tensor])[0]

# Skor > 0.7 olanları al ve maskeleri squeeze et
instance_masks_rcnn = [
    mask[0].cpu().numpy()  # Squeeze: [1, H, W] -> [H, W]
    for i, mask in enumerate(prediction["masks"])
    if prediction["scores"][i] > 0.7
]

# === Semantic Segmentation için ===
semantic_yolo = (
    np.sum(instance_masks_yolo, axis=0)
    if len(instance_masks_yolo) > 0
    else np.zeros((640, 640))
)
semantic_rcnn = (
    np.sum(instance_masks_rcnn, axis=0)
    if len(instance_masks_rcnn) > 0
    else np.zeros((640, 640))
)

# === Görselleştirme ===
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axs[0, 0].imshow(image_rgb)
axs[0, 0].set_title("YOLOv8 - Instance Segmentation")
for mask in instance_masks_yolo:
    axs[0, 0].imshow(np.ma.masked_where(mask < 0.5, mask), cmap="Greens", alpha=0.5)

axs[0, 1].imshow(image_rgb)
axs[0, 1].set_title("Mask R-CNN - Instance Segmentation")
for mask in instance_masks_rcnn:
    axs[0, 1].imshow(np.ma.masked_where(mask < 0.5, mask), cmap="Reds", alpha=0.5)

axs[1, 0].imshow(image_rgb)
axs[1, 0].set_title("YOLOv8 - Semantic Segmentation (benzeri)")
axs[1, 0].imshow(
    np.ma.masked_where(semantic_yolo < 0.5, semantic_yolo), cmap="Greens", alpha=0.5
)

axs[1, 1].imshow(image_rgb)
axs[1, 1].set_title("Mask R-CNN - Semantic Segmentation (benzeri)")
axs[1, 1].imshow(
    np.ma.masked_where(semantic_rcnn < 0.5, semantic_rcnn), cmap="Reds", alpha=0.5
)

for ax in axs.flat:
    ax.axis("off")

plt.tight_layout()
plt.show()
