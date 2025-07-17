import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor


class OneFormerSegmentationComparison:
    def __init__(self):
        """OneFormer modelini ve processor'ı yükle"""
        print("OneFormer modeli yükleniyor...")

        # COCO dataset için eğitilmiş OneFormer modeli
        self.model_name = "shi-labs/oneformer_coco_swin_large"

        # Processor ve model yükleme
        self.processor = OneFormerProcessor.from_pretrained(self.model_name)
        self.model = OneFormerForUniversalSegmentation.from_pretrained(self.model_name)

        # GPU varsa kullan
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        print(f"Model {self.device} üzerinde yüklendi")

        # COCO dataset sınıf isimleri
        self.class_names = [
            "person",
            "bicycle",
            "car",
            "motorcycle",
            "airplane",
            "bus",
            "train",
            "truck",
            "boat",
            "traffic light",
            "fire hydrant",
            "stop sign",
            "parking meter",
            "bench",
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "backpack",
            "umbrella",
            "handbag",
            "tie",
            "suitcase",
            "frisbee",
            "skis",
            "snowboard",
            "sports ball",
            "kite",
            "baseball bat",
            "baseball glove",
            "skateboard",
            "surfboard",
            "tennis racket",
            "bottle",
            "wine glass",
            "cup",
            "fork",
            "knife",
            "spoon",
            "bowl",
            "banana",
            "apple",
            "sandwich",
            "orange",
            "broccoli",
            "carrot",
            "hot dog",
            "pizza",
            "donut",
            "cake",
            "chair",
            "couch",
            "potted plant",
            "bed",
            "dining table",
            "toilet",
            "tv",
            "laptop",
            "mouse",
            "remote",
            "keyboard",
            "cell phone",
            "microwave",
            "oven",
            "toaster",
            "sink",
            "refrigerator",
            "book",
            "clock",
            "vase",
            "scissors",
            "teddy bear",
            "hair drier",
            "toothbrush",
        ]

    def load_image(self, image_path):
        """Görüntüyü yükle"""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Görüntü bulunamadı: {image_path}")

        image = Image.open(image_path).convert("RGB")
        return image

    def perform_semantic_segmentation(self, image):
        """Semantic segmentasyon gerçekleştir"""
        print("Semantic segmentasyon yapılıyor...")

        # Semantic segmentasyon için input hazırlama
        inputs = self.processor(
            image, task_inputs=["semantic"], return_tensors="pt"
        ).to(self.device)

        # Model tahminleri
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Sonuçları işleme
        predicted_semantic_map = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.size[::-1]]
        )[0]

        return predicted_semantic_map

    def perform_instance_segmentation(self, image):
        """Instance segmentasyon gerçekleştir"""
        print("Instance segmentasyon yapılıyor...")

        # Instance segmentasyon için input hazırlama
        inputs = self.processor(
            image, task_inputs=["instance"], return_tensors="pt"
        ).to(self.device)

        # Model tahminleri
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Sonuçları işleme
        predicted_instance_map = self.processor.post_process_instance_segmentation(
            outputs, target_sizes=[image.size[::-1]]
        )[0]

        return predicted_instance_map

    def visualize_comparison(self, image, semantic_map, instance_map, save_path=None):
        """Semantic ve instance segmentasyon sonuçlarını görselleştir"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Orijinal görüntü
        axes[0].imshow(image)
        axes[0].set_title("Orijinal Görüntü", fontsize=16, fontweight="bold")
        axes[0].axis("off")

        # Semantic segmentasyon - transparan overlay
        semantic_overlay = self.create_transparent_overlay(
            image, semantic_map, "semantic"
        )
        axes[1].imshow(semantic_overlay)
        axes[1].set_title("Semantic Segmentasyon", fontsize=16, fontweight="bold")
        axes[1].axis("off")

        # Instance segmentasyon - transparan overlay
        instance_overlay = self.create_transparent_overlay(
            image, instance_map, "instance"
        )
        axes[2].imshow(instance_overlay)
        axes[2].set_title("Instance Segmentasyon", fontsize=16, fontweight="bold")
        axes[2].axis("off")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Görselleştirme kaydedildi: {save_path}")

        plt.show()

    def colorize_semantic_map(self, semantic_map):
        """Semantic haritayı renklendir"""
        # Tensor'ı CPU'ya taşı ve numpy'a çevir
        semantic_np = semantic_map.cpu().numpy()

        # Rastgele renkler oluştur
        np.random.seed(42)  # Tutarlılık için
        colors = np.random.randint(0, 255, size=(len(self.class_names) + 1, 3))
        colors[0] = [0, 0, 0]  # Arka plan siyah

        colored_map = np.zeros((*semantic_np.shape, 3), dtype=np.uint8)
        for class_id in range(len(self.class_names) + 1):
            mask = semantic_np == class_id
            colored_map[mask] = colors[class_id]

        return colored_map

    def colorize_instance_map(self, instance_map):
        """Instance haritayı renklendir"""
        segmentation = instance_map["segmentation"].cpu().numpy()

        # Her instance için farklı renk
        unique_ids = np.unique(segmentation)
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(len(unique_ids), 3))

        colored_map = np.zeros((*segmentation.shape, 3), dtype=np.uint8)
        for i, instance_id in enumerate(unique_ids):
            if instance_id == 0:  # Arka plan
                continue
            mask = segmentation == instance_id
            colored_map[mask] = colors[i]

        return colored_map

    def create_transparent_overlay(self, image, segmentation_map, seg_type):
        """Orijinal görüntü üzerine transparan segmentasyon overlay'i oluştur"""
        # Görüntüyü array'e çevir
        img_array = np.array(image)

        # Alpha değeri (şeffaflık)
        alpha = 0.4  # Segmentasyon renkleri için

        if seg_type == "semantic":
            # Semantic segmentasyon renklerini oluştur
            colored_map = self.colorize_semantic_map(segmentation_map)

            # Arka plan maskesi (sınıf 0)
            semantic_np = segmentation_map.cpu().numpy()
            background_mask = semantic_np == 0

        elif seg_type == "instance":
            # Instance segmentasyon renklerini oluştur
            colored_map = self.colorize_instance_map(segmentation_map)

            # Arka plan maskesi (instance 0)
            segmentation_np = segmentation_map["segmentation"].cpu().numpy()
            background_mask = segmentation_np == 0

        # Overlay oluştur
        overlay = img_array.astype(np.float32)

        # Sadece segmentasyon bulunan alanlara renk uygula
        for i in range(3):  # RGB kanalları
            overlay[:, :, i] = np.where(
                background_mask,
                img_array[:, :, i],  # Arka plan - orijinal görüntü
                alpha * colored_map[:, :, i]
                + (1 - alpha) * img_array[:, :, i],  # Segmentasyon - transparan mix
            )

        return overlay.astype(np.uint8)

    def analyze_results(self, semantic_map, instance_map):
        """Sonuçları analiz et"""
        print("\n=== SEGMENTASYON ANALİZİ ===")

        # Semantic analiz - tensor'ı CPU'ya taşı
        semantic_np = semantic_map.cpu().numpy()
        unique_classes = np.unique(semantic_np)
        print("\nSemantic Segmentasyon:")
        print(f"- Toplam sınıf sayısı: {len(unique_classes)}")
        print("- Tespit edilen sınıflar:")
        for class_id in unique_classes:
            if class_id > 0 and class_id <= len(self.class_names):
                pixel_count = (semantic_np == class_id).sum()
                percentage = (pixel_count / semantic_np.size) * 100
                print(
                    f"  • {self.class_names[class_id-1]}: {pixel_count} piksel "
                    f"({percentage:.1f}%)"
                )

        # Instance analiz - tensor'ı CPU'ya taşı
        segmentation = instance_map["segmentation"].cpu().numpy()
        unique_instances = np.unique(segmentation)
        print("\nInstance Segmentasyon:")
        # -1 arka plan için
        print(f"- Toplam instance sayısı: {len(unique_instances) - 1}")

        if "segments_info" in instance_map:
            print("- Instance detayları:")
            for segment in instance_map["segments_info"]:
                class_id = segment["label_id"]
                if class_id <= len(self.class_names):
                    area = segment["area"]
                    print(f"  • {self.class_names[class_id]}: {area} piksel")

    def run_comparison(self, image_path, save_path=None):
        """Tam karşılaştırma işlemini çalıştır"""
        try:
            # Görüntüyü yükle
            image = self.load_image(image_path)
            print(f"Görüntü yüklendi: {image.size}")

            # Segmentasyon işlemleri
            semantic_map = self.perform_semantic_segmentation(image)
            instance_map = self.perform_instance_segmentation(image)

            # Görselleştirme
            self.visualize_comparison(image, semantic_map, instance_map, save_path)

            # Analiz
            self.analyze_results(semantic_map, instance_map)

            print("\nKarşılaştırma tamamlandı!")

        except Exception as e:
            print(f"Hata oluştu: {e}")


# Kullanım örneği
if __name__ == "__main__":
    # Segmentasyon karşılaştırıcısını başlat
    comparator = OneFormerSegmentationComparison()

    # test.png dosyasını karşılaştır
    image_path = "test.png"
    save_path = "output.png"

    # Karşılaştırmayı çalıştır
    comparator.run_comparison(image_path, save_path)
