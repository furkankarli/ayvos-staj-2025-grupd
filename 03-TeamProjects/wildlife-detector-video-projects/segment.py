import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

# Cihaz seçimi: M1/M2 için "mps", varsa "cuda", yoksa "cpu"
device = "cpu"

print(f"[INFO] Kullanılan cihaz: {device}")


# 1. SAM modelini yükle
def load_sam_model(checkpoint_path, model_type="vit_h"):
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device)
    return SamAutomaticMaskGenerator(sam)


# 2. Görseli oku ve maskeleri oluştur
def generate_masks(image_path, mask_generator):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image)
    return image, masks


# 3. Maske çıktısını görselleştir
def show_masks_on_image(image, masks, save_path=None):
    output = image.copy()
    for mask in masks:
        m = mask["segmentation"]
        color = np.random.randint(0, 255, (3,), dtype=np.uint8)
        output[m] = color

    if save_path:
        cv2.imwrite(save_path, cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
    else:
        plt.imshow(output)
        plt.axis("off")
        plt.show()


# 4. Ana işlem
def main():
    checkpoint_path = "models/sam_vit_h_4b8939.pth"  # İndirdiğin model dosyası
    model_type = "vit_h"

    mask_generator = load_sam_model(checkpoint_path, model_type)

    input_folder = "data"
    output_folder = "outputs"
    os.makedirs(output_folder, exist_ok=True)

    for file_name in sorted(os.listdir(input_folder)):
        if file_name.endswith(".jpg"):
            image_path = os.path.join(input_folder, file_name)
            image, masks = generate_masks(image_path, mask_generator)
            save_path = os.path.join(output_folder, f"masked_{file_name}")
            show_masks_on_image(image, masks, save_path)
            print(f"{file_name} işlendi ✔️ {len(masks)} maske bulundu")


if __name__ == "__main__":
    main()
