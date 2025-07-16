import cv2
import numpy as np
from ultralytics import YOLO


def put_label(img, label):
    h, w = img.shape[:2]
    labeled_img = np.ones((h + 40, w, 3), dtype=np.uint8) * 255
    labeled_img[40:, :, :] = img
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]
    text_x = (w - text_size[0]) // 2
    text_y = 30
    cv2.putText(
        labeled_img,
        label,
        (text_x, text_y),
        font,
        font_scale,
        (0, 0, 0),
        font_thickness,
        cv2.LINE_AA,
    )
    return labeled_img


# Use your local image path here:
image_path = r"D:\PycharmProjects\Staj\ayvos-staj-2025-grupd\cat.jpg"

# Load model
model = YOLO("yolov8n-seg.pt")

# Read image
image = cv2.imread(image_path)

# Predict
results = model.predict(source=image, conf=0.5)
result = results[0]

# Extract masks and classes
masks = result.masks.data.cpu().numpy()  # shape: [N, H, W]
classes = result.boxes.cls.cpu().numpy()  # shape: [N]

print("Detected classes (class IDs):", [int(c) for c in classes])
print("Masks shape:", masks.shape)
print(f"Number of detected masks: {len(masks)}")

if len(masks) > 0:
    print("First mask min/max values:", masks[0].min(), masks[0].max())

# Option 1: Use semantic mask for a detected class (e.g. class ID 15)
target_class = 15  # change if you want

semantic_mask = np.zeros_like(masks[0], dtype=bool)
for i, cls_id in enumerate(classes):
    if int(cls_id) == target_class:
        semantic_mask |= masks[i].astype(bool)

if semantic_mask.sum() == 0:
    print(
        f"No masks found for class id {target_class}. Using all detected masks instead."
    )
    # Option 2: fallback to all masks combined
    semantic_mask = np.zeros_like(masks[0], dtype=bool)
    for i in range(len(classes)):
        semantic_mask |= masks[i].astype(bool)

semantic_mask_uint8 = semantic_mask.astype(np.uint8) * 255
semantic_colored = np.stack([semantic_mask_uint8] * 3, axis=-1)

# Instance segmentation output
instance_image = result.plot()

# Resize images
size = (640, 640)
image_resized = cv2.resize(image, size)
instance_resized = cv2.resize(instance_image, size)
semantic_resized = cv2.resize(semantic_colored, size)

# Add labels
image_labeled = put_label(image_resized, "Original")
instance_labeled = put_label(instance_resized, "Instance")
semantic_labeled = put_label(semantic_resized, "Semantic")

# Concatenate and save
combined = cv2.hconcat([image_labeled, instance_labeled, semantic_labeled])
cv2.imwrite("output.jpg", combined)
print("Saved output.jpg with labeled images.")
