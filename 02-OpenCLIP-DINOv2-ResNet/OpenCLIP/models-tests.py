import time

import matplotlib.pyplot as plt
import open_clip
import pandas as pd
import torch
from PIL import Image


class ModelComparison:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models_to_test = [
            ("ViT-B-32", "openai"),
            ("ViT-B-16", "openai"),
            ("ViT-L-14", "openai"),
            ("ViT-B-32", "laion2b_s34b_b79k"),
            ("ViT-L-14", "laion2b_s34b_b79k"),
            ("RN50", "openai"),
            ("RN101", "openai"),
            ("ConvNeXT-Base", "laion2b_s13b_b82k"),
        ]

    def benchmark_models(self, test_image_path, class_names):
        """Farklı modelleri hız ve accuracy açısından karşılaştır"""
        results = []

        for model_name, pretrained in self.models_to_test:
            try:
                print(f"Testing {model_name} with {pretrained}...")

                # Model yükle
                start_time = time.time()
                model, _, preprocess = open_clip.create_model_and_transforms(
                    model_name, pretrained=pretrained
                )
                tokenizer = open_clip.get_tokenizer(model_name)
                model.to(self.device)
                load_time = time.time() - start_time

                # Test görüntüsü
                image = Image.open(test_image_path).convert("RGB")
                image_input = preprocess(image).unsqueeze(0).to(self.device)

                # Text tokenize
                text_tokens = tokenizer(class_names).to(self.device)

                # Inference hızı test et
                model.eval()
                with torch.no_grad():
                    # Warmup
                    for _ in range(5):
                        _ = model.encode_image(image_input)
                        _ = model.encode_text(text_tokens)

                    # Gerçek test
                    start_time = time.time()
                    for _ in range(20):
                        image_features = model.encode_image(image_input)
                        text_features = model.encode_text(text_tokens)
                        similarity = (image_features @ text_features.T).squeeze(0)
                        probs = torch.softmax(similarity, dim=0)
                    inference_time = (time.time() - start_time) / 20

                # Model bilgileri
                total_params = sum(p.numel() for p in model.parameters())

                results.append(
                    {
                        "Model": f"{model_name}_{pretrained}",
                        "Load Time (s)": load_time,
                        "Inference Time (ms)": inference_time * 1000,
                        "Parameters (M)": total_params / 1_000_000,
                        "Top Prediction": class_names[torch.argmax(probs).item()],
                        "Confidence": torch.max(probs).item(),
                    }
                )

                # Bellek temizle
                del model, preprocess, tokenizer
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"Error with {model_name}_{pretrained}: {e}")
                continue

        return pd.DataFrame(results)


# Kullanım
def test_model_comparison():
    # Test edilecek sınıflar
    car_classes = [
        "a photo of a BMW X3",
        "a photo of an Audi A4",
        "a photo of a Ford Focus",
        "a photo of a Tesla S",
    ]

    comparator = ModelComparison()

    test_image = "zero-shot/datasetv2/bmw-x3/0000.jpg"

    results_df = comparator.benchmark_models(test_image, car_classes)
    print(results_df.to_string(index=False))

    # Sonuçları kaydet
    results_df.to_csv("model_comparison_results.csv", index=False)

    # Görselleştirme
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Inference Time
    axes[0, 0].bar(range(len(results_df)), results_df["Inference Time (ms)"])
    axes[0, 0].set_title("Inference Time (ms)")
    axes[0, 0].set_xticks(range(len(results_df)))
    axes[0, 0].set_xticklabels(results_df["Model"], rotation=45)

    # Parameters
    axes[0, 1].bar(range(len(results_df)), results_df["Parameters (M)"])
    axes[0, 1].set_title("Model Size (Million Parameters)")
    axes[0, 1].set_xticks(range(len(results_df)))
    axes[0, 1].set_xticklabels(results_df["Model"], rotation=45)

    # Confidence
    axes[1, 0].bar(range(len(results_df)), results_df["Confidence"])
    axes[1, 0].set_title("Prediction Confidence")
    axes[1, 0].set_xticks(range(len(results_df)))
    axes[1, 0].set_xticklabels(results_df["Model"], rotation=45)

    # Load Time
    axes[1, 1].bar(range(len(results_df)), results_df["Load Time (s)"])
    axes[1, 1].set_title("Model Load Time (s)")
    axes[1, 1].set_xticks(range(len(results_df)))
    axes[1, 1].set_xticklabels(results_df["Model"], rotation=45)

    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    test_model_comparison()
