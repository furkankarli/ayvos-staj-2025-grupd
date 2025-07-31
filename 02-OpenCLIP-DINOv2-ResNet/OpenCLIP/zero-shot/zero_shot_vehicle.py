import json
import os

import matplotlib.pyplot as plt
import numpy as np
import open_clip
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm


class CarZeroShotClassifier:
    def __init__(self, model_name="ViT-B-32", pretrained="openai"):
        """
        OpenCLIP ile zero-shot araç sınıflandırıcı

        Args:
            model_name: Kullanılacak model (ViT-B-32, ViT-L-14, etc.)
            pretrained: Pretrained weights ('openai', 'laion2b_s34b_b79k', etc.)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Model ve tokenizer yükleme
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Sınıf isimleri
        self.class_names = [
            "a photo of an Alfa Romeo Giulia",
            "a photo of an Audi A4",
            "a photo of an Audi A6",
            "a photo of a BMW 3 Series",
            "a photo of a BMW X3",
            "a photo of a Citroen C3",
            "a photo of a Citroen C4 Grand Picasso",
            "a photo of a Dacia Logan",
            "a photo of a Dacia Spring",
            "a photo of a Fiat Bravo",
            "a photo of a Ford Fiesta",
            "a photo of a Ford Focus",
            "a photo of a Ford Fusion",
            "a photo of a Ford Mondeo",
            "a photo of a Ford Transit",
            "a photo of a Honda Civic",
            "a photo of a Hyundai i30",
            "a photo of a Kia Sportage",
            "a photo of a Maserati Levante",
            "a photo of a Mazda 2",
            "a photo of a Mini Countryman",
            "a photo of a Mitsubishi L200",
            "a photo of an Opel Astra",
            "a photo of an Opel Corsa",
            "a photo of an Opel Meriva",
            "a photo of a Peugeot 208",
            "a photo of a Peugeot 3008",
            "a photo of a Renault Captur",
            "a photo of a Seat Ibiza",
            "a photo of a Seat Leon",
            "a photo of a Skoda Fabia",
            "a photo of a Skoda Octavia",
            "a photo of a Skoda Superb",
            "a photo of a Smart Forfour",
            "a photo of a Smart Fortwo",
            "a photo of a Suzuki SX4 S-Cross",
            "a photo of a Suzuki Vitara",
            "a photo of a Tesla S",
            "a photo of a Toyota C-HR",
            "a photo of a Toyota Corolla",
            "a photo of a Toyota Yaris",
            "a photo of a Volkswagen Golf",
            "a photo of a Volkswagen Passat",
            "a photo of a Volkswagen Polo",
        ]

        # Basit sınıf isimleri (klasör isimleri için)
        self.simple_class_names = [
            "alfa-romeo-giulia",
            "audi-a4",
            "audi-a6",
            "bmw-3-series",
            "bmw-x3",
            "citroen-c3",
            "citroen-c4-grand-picasso",
            "dacia-logan",
            "dacia-spring",
            "fiat-bravo",
            "ford-fiesta",
            "ford-focus",
            "ford-fusion",
            "ford-mondeo",
            "ford-transit",
            "honda-civic",
            "hyundai-i30",
            "kia-sportage",
            "maserati-levante",
            "mazda-2",
            "mini-countryman",
            "mitsubishi-l200",
            "opel-astra",
            "opel-corsa",
            "opel-meriva",
            "peugeot-208",
            "peugeot-3008",
            "renault-captur",
            "seat-ibiza",
            "seat-leon",
            "skoda-fabia",
            "skoda-octavia",
            "skoda-superb",
            "smart-forfour",
            "smart-fortwo",
            "suzuki-sx4-s-cross",
            "suzuki-vitara",
            "tesla-s",
            "toyota-c-hr",
            "toyota-corolla",
            "toyota-yaris",
            "volkswagen-golf",
            "volkswagen-passat",
            "volkswagen-polo",
        ]

        # Text embeddings'i önceden hesapla
        self._precompute_text_embeddings()

    def _precompute_text_embeddings(self):
        """Text embeddings'i önceden hesapla"""
        with torch.no_grad():
            text_tokens = self.tokenizer(self.class_names).to(self.device)
            self.text_features = self.model.encode_text(text_tokens)
            self.text_features = self.text_features / self.text_features.norm(
                dim=-1, keepdim=True
            )

    def predict_single_image(self, image_path):
        """Tek bir görüntü için tahmin yap"""
        try:
            # Görüntüyü yükle ve işle
            image = Image.open(image_path).convert("RGB")
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                # Image embeddings hesapla
                image_features = self.model.encode_image(image_input)
                image_features = image_features / image_features.norm(
                    dim=-1, keepdim=True
                )

                # Similarity hesapla
                similarity = (image_features @ self.text_features.T).squeeze(0)
                probs = torch.softmax(similarity, dim=0)

                # En yüksek olasılıklı sınıfı döndür
                predicted_idx = torch.argmax(probs).item()
                confidence = probs[predicted_idx].item()

                return predicted_idx, confidence, probs.cpu().numpy()

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None, 0.0, None

    def evaluate_dataset(
        self, dataset_path, sample_angles=None, max_images_per_class=None
    ):
        """
        Tüm dataset üzerinde değerlendirme yap

        Args:
            dataset_path: Dataset ana klasörü yolu
            sample_angles: Test edilecek açılar listesi (None ise hepsi)
            max_images_per_class: Her sınıf için maksimum test görüntüsü sayısı
        """
        predictions = []
        true_labels = []
        image_paths = []
        confidences = []

        print("Evaluating dataset...")

        for class_idx, class_name in enumerate(tqdm(self.simple_class_names)):
            class_folder = os.path.join(dataset_path, class_name)

            if not os.path.exists(class_folder):
                print(f"Warning: Class folder not found: {class_folder}")
                continue

            # Görüntü dosyalarını al
            image_files = []
            if sample_angles is None:
                # Tüm açılar
                for angle in range(0, 366, 6):  # 0, 6, 12, ..., 360
                    img_file = f"{angle:04d}.jpg"
                    img_path = os.path.join(class_folder, img_file)
                    if os.path.exists(img_path):
                        image_files.append(img_path)
            else:
                # Belirli açılar
                for angle in sample_angles:
                    img_file = f"{angle:04d}.jpg"
                    img_path = os.path.join(class_folder, img_file)
                    if os.path.exists(img_path):
                        image_files.append(img_path)

            # Maksimum görüntü sayısını sınırla
            if max_images_per_class and len(image_files) > max_images_per_class:
                image_files = image_files[:max_images_per_class]

            # Her görüntü için tahmin yap
            for img_path in image_files:
                pred_idx, confidence, _ = self.predict_single_image(img_path)

                if pred_idx is not None:
                    predictions.append(pred_idx)
                    true_labels.append(class_idx)
                    confidences.append(confidence)
                    image_paths.append(img_path)

        return predictions, true_labels, confidences, image_paths

    def calculate_metrics(self, predictions, true_labels):
        """Metrikleri hesapla"""
        if len(predictions) == 0:
            return {"overall_accuracy": 0.0, "class_accuracies": {}, "num_samples": 0}

        accuracy = accuracy_score(true_labels, predictions)

        # Her sınıf için accuracy
        class_accuracies = {}
        for i, class_name in enumerate(self.simple_class_names):
            class_mask = np.array(true_labels) == i
            if np.sum(class_mask) > 0:
                class_pred = np.array(predictions)[class_mask]
                class_true = np.array(true_labels)[class_mask]
                class_acc = accuracy_score(class_true, class_pred)
                class_accuracies[class_name] = class_acc

        return {
            "overall_accuracy": accuracy,
            "class_accuracies": class_accuracies,
            "num_samples": len(predictions),
        }

    def plot_confusion_matrix(self, predictions, true_labels, save_path=None):
        """Confusion matrix çiz"""
        if len(predictions) == 0:
            print("No predictions to plot confusion matrix")
            return

        cm = confusion_matrix(true_labels, predictions)

        plt.figure(figsize=(15, 12))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=[
                name.replace("a photo of ", "").replace("an ", "")
                for name in self.class_names
            ],
            yticklabels=[
                name.replace("a photo of ", "").replace("an ", "")
                for name in self.class_names
            ],
        )
        plt.title("Confusion Matrix - Zero-Shot Car Classification")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

    def analyze_angle_performance(self, dataset_path, test_angles=None):
        """Farklı açılarda performans analizi"""
        if test_angles is None:
            test_angles = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]

        angle_results = {}

        for angle in tqdm(test_angles, desc="Testing angles"):
            predictions, true_labels, confidences, _ = self.evaluate_dataset(
                dataset_path, sample_angles=[angle]
            )

            if predictions:
                accuracy = accuracy_score(true_labels, predictions)
                avg_confidence = np.mean(confidences)
                angle_results[angle] = {
                    "accuracy": accuracy,
                    "avg_confidence": avg_confidence,
                    "num_samples": len(predictions),
                }

        return angle_results

    def save_results(self, results, save_path):
        """Sonuçları JSON olarak kaydet"""
        # NumPy array'leri Python listelerine çevir
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                serializable_results[key] = value.tolist()
            elif isinstance(value, dict):
                serializable_results[key] = {}
                for k, v in value.items():
                    if isinstance(v, (np.integer, np.floating)):
                        serializable_results[key][k] = float(v)
                    else:
                        serializable_results[key][k] = v
            else:
                serializable_results[key] = value

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)


# Kullanım örneği
def main():
    # Classifier'ı başlat
    classifier = CarZeroShotClassifier(model_name="ViT-B-32", pretrained="openai")

    # Dataset yolu
    dataset_path = "datasetv2"  # Ana dataset klasörü

    print("Starting zero-shot evaluation...")

    # 1. Tüm dataset üzerinde değerlendirme (sample alarak)
    predictions, true_labels, confidences, image_paths = classifier.evaluate_dataset(
        dataset_path,
        sample_angles=[0, 90, 180, 270],  # 4 ana açı
        max_images_per_class=20,  # Her sınıf için maksimum 20 görüntü
    )

    # 2. Metrikleri hesapla
    metrics = classifier.calculate_metrics(predictions, true_labels)

    print(f"\nOverall Accuracy: {metrics['overall_accuracy']:.4f}")
    print(f"Number of test samples: {metrics['num_samples']}")

    # 3. En iyi ve en kötü performans gösteren sınıflar
    class_accs = metrics["class_accuracies"]
    best_classes = sorted(class_accs.items(), key=lambda x: x[1], reverse=True)[:5]
    worst_classes = sorted(class_accs.items(), key=lambda x: x[1])[:5]

    print("\nTop 5 best performing classes:")
    for class_name, acc in best_classes:
        print(f"  {class_name}: {acc:.4f}")

    print("\nTop 5 worst performing classes:")
    for class_name, acc in worst_classes:
        print(f"  {class_name}: {acc:.4f}")

    # 4. Confusion matrix çiz (eğer veri varsa)
    if predictions:
        classifier.plot_confusion_matrix(
            predictions, true_labels, "confusion_matrix.png"
        )

    # 5. Açı analizi (opsiyonel - eğer veri varsa)
    if predictions:
        print("\nAnalyzing performance across different angles...")
        angle_results = classifier.analyze_angle_performance(
            dataset_path, test_angles=[0, 45, 90, 135, 180, 225, 270, 315]
        )

        print("\nAngle-wise performance:")
        for angle, result in angle_results.items():
            print(
                f"  {angle}°: Accuracy={result['accuracy']:.4f}, "
                f"Confidence={result['avg_confidence']:.4f}"
            )
    else:
        print("\nNo data found for angle analysis")
        angle_results = {}

    # 6. Sonuçları kaydet
    all_results = {
        "metrics": metrics,
        "angle_results": angle_results,
        "model_info": {
            "model_name": "ViT-B-32",
            "pretrained": "openai",
            "num_classes": len(classifier.class_names),
        },
    }

    classifier.save_results(all_results, "zeroshot_results.json")
    print("\nResults saved to 'zeroshot_results.json'")


if __name__ == "__main__":
    main()
