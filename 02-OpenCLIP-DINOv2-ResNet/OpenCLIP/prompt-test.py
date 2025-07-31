import matplotlib.pyplot as plt
import numpy as np
import open_clip
import pandas as pd
import seaborn as sns
import torch
from PIL import Image


class PromptEngineeringTest:
    def __init__(self, model_name="ViT-B-32", pretrained="openai"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.to(self.device)
        self.model.eval()

    def test_prompt_templates(self, image_path, base_class_name):
        """Farklı prompt template'lerini test et"""

        # Farklı prompt şablonları
        prompt_templates = {
            "basic": f"a photo of a {base_class_name}",
            "detailed": f"a high quality photo of a {base_class_name}",
            "contextual": f"a photo of a {base_class_name} on a road",
            "professional": f"a professional photograph of a {base_class_name}",
            "descriptive": f"a clear photo of a {base_class_name} car",
            "artistic": f"an artistic photo of a {base_class_name}",
            "realistic": f"a realistic photo of a {base_class_name} vehicle",
            "studio": f"a studio photo of a {base_class_name}",
            "outdoor": f"an outdoor photo of a {base_class_name}",
            "side_view": f"a side view photo of a {base_class_name}",
            "front_view": f"a front view photo of a {base_class_name}",
            "angle_view": f"an angled view photo of a {base_class_name}",
        }

        # Distractor sınıflar (karışıklık yaratması için)
        distractors = [
            "a photo of a Toyota Corolla",
            "a photo of a Honda Civic",
            "a photo of a Ford Focus",
            "a photo of a Volkswagen Golf",
        ]

        results = {}

        # Görüntüyü yükle
        image = Image.open(image_path).convert("RGB")
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            for template_name, prompt in prompt_templates.items():
                # Test edilecek tüm sınıflar
                all_classes = [prompt] + distractors

                # Text embeddings
                text_tokens = self.tokenizer(all_classes).to(self.device)
                text_features = self.model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                # Similarity hesapla
                similarity = (image_features @ text_features.T).squeeze(0)
                probs = torch.softmax(similarity, dim=0)

                # Doğru sınıfın skorunu kaydet (ilk sınıf)
                target_confidence = probs[0].item()
                target_rank = (
                    torch.argsort(probs, descending=True).tolist().index(0) + 1
                )

                results[template_name] = {
                    "confidence": target_confidence,
                    "rank": target_rank,
                    "prompt": prompt,
                }

        return results

    def test_ensemble_prompts(self, image_path, base_class_name):
        """Birden fazla prompt'un ensemble'ını test et"""

        ensemble_prompts = [
            f"a photo of a {base_class_name}",
            f"a clear image of a {base_class_name}",
            f"a {base_class_name} car",
            f"a picture of a {base_class_name} vehicle",
        ]

        distractors = [
            "a photo of a Toyota Corolla",
            "a photo of a Honda Civic",
            "a photo of a Ford Focus",
        ]

        # Görüntüyü yükle
        image = Image.open(image_path).convert("RGB")
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Her prompt için sonuçları topla
            ensemble_scores = []

            for prompt in ensemble_prompts:
                all_classes = [prompt] + distractors
                text_tokens = self.tokenizer(all_classes).to(self.device)
                text_features = self.model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)

                similarity = (image_features @ text_features.T).squeeze(0)
                probs = torch.softmax(similarity, dim=0)
                ensemble_scores.append(probs.cpu().numpy())

            # Ensemble - ortalama al
            ensemble_avg = np.mean(ensemble_scores, axis=0)

            # Tek prompt vs ensemble karşılaştırması
            single_prompt = ensemble_prompts[0]
            all_classes = [single_prompt] + distractors
            text_tokens = self.tokenizer(all_classes).to(self.device)
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            similarity = (image_features @ text_features.T).squeeze(0)
            single_probs = torch.softmax(similarity, dim=0).cpu().numpy()

            return {
                "single_confidence": single_probs[0],
                "ensemble_confidence": ensemble_avg[0],
                "single_rank": np.argsort(single_probs)[::-1].tolist().index(0) + 1,
                "ensemble_rank": np.argsort(ensemble_avg)[::-1].tolist().index(0) + 1,
            }

    def test_multilingual_prompts(self, image_path, class_translations):
        """Çok dilli prompt'ları test et"""

        distractors_multilang = {
            "english": ["a photo of a Toyota Corolla", "a photo of a Honda Civic"],
            "turkish": ["bir Toyota Corolla fotoğrafı", "bir Honda Civic fotoğrafı"],
            "german": [
                "ein Foto von einem Toyota Corolla",
                "ein Foto von einem Honda Civic",
            ],
            "french": ["une photo d'une Toyota Corolla", "une photo d'une Honda Civic"],
        }

        # Görüntüyü yükle
        image = Image.open(image_path).convert("RGB")
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)

        results = {}

        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            for lang, prompt in class_translations.items():
                if lang in distractors_multilang:
                    all_classes = [prompt] + distractors_multilang[lang]

                    text_tokens = self.tokenizer(all_classes).to(self.device)
                    text_features = self.model.encode_text(text_tokens)
                    text_features = text_features / text_features.norm(
                        dim=-1, keepdim=True
                    )

                    similarity = (image_features @ text_features.T).squeeze(0)
                    probs = torch.softmax(similarity, dim=0)

                    results[lang] = {
                        "confidence": probs[0].item(),
                        "rank": torch.argsort(probs, descending=True).tolist().index(0)
                        + 1,
                        "prompt": prompt,
                    }

        return results


# Kapsamlı test fonksiyonu
def comprehensive_prompt_test():
    """Tüm prompt testlerini çalıştır"""

    tester = PromptEngineeringTest()

    # Test görüntüsü ve sınıf
    test_image = "zero-shot/datasetv2/bmw-x3/0000.jpg"
    target_class = "BMW X3"

    print("1. Testing different prompt templates...")
    template_results = tester.test_prompt_templates(test_image, target_class)

    # Sonuçları DataFrame'e çevir
    template_df = pd.DataFrame.from_dict(template_results, orient="index")
    template_df = template_df.sort_values("confidence", ascending=False)

    print("\nPrompt Template Results:")
    print(template_df.to_string())

    # Görselleştirme
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    sns.barplot(data=template_df.reset_index(), x="confidence", y="index")
    plt.title("Confidence by Prompt Template")
    plt.xlabel("Confidence Score")

    plt.subplot(2, 2, 2)
    sns.barplot(data=template_df.reset_index(), x="rank", y="index")
    plt.title("Rank by Prompt Template (Lower is Better)")
    plt.xlabel("Rank")

    print("\n2. Testing ensemble prompts...")
    ensemble_results = tester.test_ensemble_prompts(test_image, target_class)

    print(f"Single Prompt Confidence: {ensemble_results['single_confidence']:.4f}")
    print(f"Ensemble Confidence: {ensemble_results['ensemble_confidence']:.4f}")
    improvement = (
        ensemble_results["ensemble_confidence"] - ensemble_results["single_confidence"]
    )
    print(f"Improvement: {improvement:.4f}")

    # Ensemble comparison
    plt.subplot(2, 2, 3)
    methods = ["Single Prompt", "Ensemble"]
    confidences = [
        ensemble_results["single_confidence"],
        ensemble_results["ensemble_confidence"],
    ]
    plt.bar(methods, confidences)
    plt.title("Single vs Ensemble Prompts")
    plt.ylabel("Confidence")

    # Multilingual test
    print("\n3. Testing multilingual prompts...")
    multilang_prompts = {
        "english": f"a photo of a {target_class}",
        "turkish": f"bir {target_class} fotoğrafı",
        "german": f"ein Foto von einem {target_class}",
        "french": f"une photo d'un {target_class}",
    }

    multilang_results = tester.test_multilingual_prompts(test_image, multilang_prompts)

    if multilang_results:
        multilang_df = pd.DataFrame.from_dict(multilang_results, orient="index")
        print("\nMultilingual Results:")
        print(multilang_df.to_string())

        plt.subplot(2, 2, 4)
        sns.barplot(data=multilang_df.reset_index(), x="confidence", y="index")
        plt.title("Confidence by Language")
        plt.xlabel("Confidence Score")

    plt.tight_layout()
    plt.savefig("prompt_engineering_results.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Sonuçları kaydet
    template_df.to_csv("prompt_template_results.csv")

    with open("ensemble_results.txt", "w") as f:
        f.write(
            f"Single Prompt Confidence: {ensemble_results['single_confidence']:.4f}\n"
        )
        f.write(f"Ensemble Confidence: {ensemble_results['ensemble_confidence']:.4f}\n")
        improvement = (
            ensemble_results["ensemble_confidence"]
            - ensemble_results["single_confidence"]
        )
        f.write(f"Improvement: {improvement:.4f}\n")


if __name__ == "__main__":
    comprehensive_prompt_test()
