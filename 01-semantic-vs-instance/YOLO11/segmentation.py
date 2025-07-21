import cv2
import matplotlib.pyplot as plt
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


def run_segmentation(image_path, model_path="yolo11x-seg.pt"):
    model = YOLO(model_path)

    image = cv2.imread(image_path)

    results = model.predict(image, conf=0.85)
    result = results[0]

    instance_image = result.plot()
    masks = result.masks.data.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy()

    semantic_mask = np.zeros_like(masks[0], dtype=bool)
    for i, cls_id in enumerate(classes):
        semantic_mask |= masks[i].astype(bool)

    semantic_mask_uint8 = semantic_mask.astype(np.uint8) * 255
    semantic_colored = np.stack([semantic_mask_uint8] * 3, axis=-1)

    size = (640, 640)
    image_resized = cv2.resize(image, size)
    instance_resized = cv2.resize(instance_image, size)
    semantic_resized = cv2.resize(semantic_colored, size)

    image_labeled = put_label(image_resized, "Original")
    instance_labeled = put_label(instance_resized, "Instance")
    semantic_labeled = put_label(semantic_resized, "Semantic")

    combined = cv2.hconcat([image_labeled, instance_labeled, semantic_labeled])

    output_path = "output.jpg"
    cv2.imwrite(output_path, combined)

    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Instance Segmentation")
    plt.imshow(cv2.cvtColor(instance_resized, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Semantic Segmentation")
    plt.imshow(cv2.cvtColor(semantic_resized, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    image_path = "test.png"
    run_segmentation(image_path)
