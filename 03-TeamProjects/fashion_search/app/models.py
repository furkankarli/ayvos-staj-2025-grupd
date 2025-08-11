import os
from typing import List, Optional, Tuple

import faiss
import numpy as np
import open_clip
import torch
from app.utils import Config, logger
from PIL import Image
from torchvision import datasets, transforms
from tqdm import tqdm


class FashionMNISTManager:
    """Manages the Fashion-MNIST dataset."""

    def __init__(self, data_path: str = None):
        self.data_path = data_path or Config.DATA_PATH
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )

    def download_dataset(self) -> datasets.FashionMNIST:
        """Downloads the Fashion-MNIST dataset."""
        logger.info(f"Downloading Fashion-MNIST to: {self.data_path}")

        try:
            dataset = datasets.FashionMNIST(
                root=self.data_path, train=True, download=True, transform=self.transform
            )
            logger.info(f"Dataset downloaded: {len(dataset)} samples")
            return dataset

        except Exception as e:
            logger.error(f"Dataset download error: {str(e)}")
            raise


class OpenCLIPManager:
    """Manages the OpenCLIP model."""

    def __init__(self):
        self.model = None
        self.preprocess = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = Config.MODEL_NAME
        self.pretrained = Config.MODEL_PRETRAINED

    def load_model(self):
        """Loads the OpenCLIP model."""
        logger.info(f"Loading OpenCLIP model: {self.model_name}")

        try:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                self.model_name, pretrained=self.pretrained
            )
            self.model = self.model.to(self.device)
            self.model.eval()

            logger.info(f"Model loaded. Device: {self.device}")
            return True

        except Exception as e:
            logger.error(f"Model loading error: {str(e)}")
            raise

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocesses the image for the model."""
        try:
            image_tensor = self.preprocess(image).unsqueeze(0)
            return image_tensor.to(self.device)

        except Exception as e:
            logger.error(f"Image preprocessing error: {str(e)}")
            raise

    def encode_image(self, image_tensor: torch.Tensor) -> np.ndarray:
        """Extracts an embedding from the image."""
        try:
            with torch.no_grad():
                image_features = self.model.encode_image(image_tensor)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                embedding = image_features.cpu().numpy()

            return embedding[0]

        except Exception as e:
            logger.error(f"Embedding extraction error: {str(e)}")
            raise


class FAISSManager:
    """Manages the FAISS index."""

    def __init__(self, embeddings_path: str = None):
        self.embeddings_path = embeddings_path or Config.EMBEDDINGS_PATH
        self.index = None
        self.embeddings = None
        self.labels = None
        self.dimension = 512  # OpenCLIP ViT-B-32 dimension

    def create_index(self, embeddings: np.ndarray):
        """Creates a FAISS index."""
        logger.info(f"Creating FAISS index. Shape: {embeddings.shape}")

        try:
            self.index = faiss.IndexFlatIP(self.dimension)
            self.index.add(embeddings)
            self.embeddings = embeddings
            logger.info(f"Index created. Total vectors: {self.index.ntotal}")

        except Exception as e:
            logger.error(f"Index creation error: {str(e)}")
            raise

    def save_index(self, filename: str = "faiss.index"):
        """Saves the index to disk."""
        try:
            os.makedirs(self.embeddings_path, exist_ok=True)
            filepath = os.path.join(self.embeddings_path, filename)
            faiss.write_index(self.index, filepath)
            logger.info(f"Index saved: {filepath}")

        except Exception as e:
            logger.error(f"Index saving error: {str(e)}")
            raise

    def load_index(self, filename: str = "faiss.index") -> bool:
        """Loads the index from disk."""
        try:
            filepath = os.path.join(self.embeddings_path, filename)
            if not os.path.exists(filepath):
                logger.warning(f"Index file not found: {filepath}")
                return False

            self.index = faiss.read_index(filepath)
            logger.info(f"Index loaded: {filepath}")
            logger.info(f"Total vectors: {self.index.ntotal}")
            return True

        except Exception as e:
            logger.error(f"Index loading error: {str(e)}")
            return False

    def search(
        self, query_embedding: np.ndarray, k: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Searches for the k most similar vectors."""
        try:
            query_embedding = query_embedding.reshape(1, -1)
            distances, indices = self.index.search(query_embedding, k)
            return distances[0], indices[0]

        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            raise

    def save_embeddings(self, embeddings: np.ndarray, filename: str = "embeddings.npy"):
        """Saves embeddings to disk."""
        try:
            os.makedirs(self.embeddings_path, exist_ok=True)
            filepath = os.path.join(self.embeddings_path, filename)
            np.save(filepath, embeddings)
            logger.info(f"Embeddings saved: {filepath}")

        except Exception as e:
            logger.error(f"Embeddings saving error: {str(e)}")
            raise

    def load_embeddings(self, filename: str = "embeddings.npy") -> Optional[np.ndarray]:
        """Loads embeddings from disk."""
        try:
            filepath = os.path.join(self.embeddings_path, filename)
            if not os.path.exists(filepath):
                logger.warning(f"Embeddings file not found: {filepath}")
                return None

            embeddings = np.load(filepath)
            logger.info(f"Embeddings loaded: {filepath}")
            logger.info(f"Embeddings shape: {embeddings.shape}")
            return embeddings

        except Exception as e:
            logger.error(f"Embeddings loading error: {str(e)}")
            return None


class FashionSearchEngine:
    """The main search engine class."""

    def __init__(self):
        self.data_manager = FashionMNISTManager()
        self.model_manager = OpenCLIPManager()
        self.faiss_manager = FAISSManager()
        self.dataset = None
        self.is_initialized = False

    def initialize(self) -> bool:
        """Initializes all components."""
        try:
            logger.info("Initializing search engine...")

            self.model_manager.load_model()
            self.dataset = self.data_manager.download_dataset()

            embeddings = self.faiss_manager.load_embeddings()

            if embeddings is None:
                logger.info("Computing embeddings...")
                embeddings = self._compute_embeddings()
                self.faiss_manager.save_embeddings(embeddings)
                self.faiss_manager.create_index(embeddings)
                self.faiss_manager.save_index()
            else:
                if not self.faiss_manager.load_index():
                    self.faiss_manager.create_index(embeddings)
                    self.faiss_manager.save_index()

            self.is_initialized = True
            logger.info("Search engine initialized successfully!")
            return True

        except Exception as e:
            logger.error(f"Search engine initialization error: {str(e)}")
            return False

    def _compute_embeddings(self, batch_size: int = 32) -> np.ndarray:
        """Computes embeddings for the entire dataset."""
        logger.info("Starting embedding computation...")

        try:
            dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=Config.MAX_WORKERS,
            )

            embeddings = []
            labels = []

            for batch_images, batch_labels in tqdm(
                dataloader, desc="Computing embeddings"
            ):
                batch_images_rgb = batch_images.repeat(1, 3, 1, 1)

                for i in range(batch_images_rgb.shape[0]):
                    image_tensor = batch_images_rgb[i : i + 1]
                    image_pil = transforms.ToPILImage()(image_tensor[0])
                    processed_image = self.model_manager.preprocess_image(image_pil)
                    embedding = self.model_manager.encode_image(processed_image)

                    embeddings.append(embedding)
                    labels.append(batch_labels[i].item())

            embeddings = np.array(embeddings)
            labels = np.array(labels)

            logger.info(f"Embedding computation finished. Shape: {embeddings.shape}")

            os.makedirs(Config.EMBEDDINGS_PATH, exist_ok=True)
            np.save(os.path.join(Config.EMBEDDINGS_PATH, "labels.npy"), labels)

            return embeddings

        except Exception as e:
            logger.error(f"Embedding computation error: {str(e)}")
            raise

    def search(self, image: Image.Image, k: int = 5) -> List[dict]:
        """Finds visually similar products."""
        if not self.is_initialized:
            raise RuntimeError("Search engine is not initialized!")

        try:
            logger.info("Searching for similar products...")

            processed_image = self.model_manager.preprocess_image(image)
            query_embedding = self.model_manager.encode_image(processed_image)
            distances, indices = self.faiss_manager.search(query_embedding, k)

            results = []
            for idx, distance in zip(indices, distances):
                img, label = self.dataset[idx]
                img_numpy = img.numpy().tolist()

                results.append(
                    {
                        "index": int(idx),
                        "label": int(label),
                        "distance": float(distance),
                        "image": img_numpy,
                    }
                )

            logger.info(f"Found {len(results)} similar products")
            return results

        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            raise


# Global search engine instance
search_engine_instance = None


def get_search_engine() -> FashionSearchEngine:
    """Returns the global search engine instance (Singleton)."""
    global search_engine_instance
    if search_engine_instance is None:
        search_engine_instance = FashionSearchEngine()
        search_engine_instance.initialize()
    return search_engine_instance


# For testing
if __name__ == "__main__":
    try:
        engine = get_search_engine()
        test_image = Image.new("L", (28, 28), 128)

        results = engine.search(test_image, k=3)

        print("Search results:")
        for result in results:
            print(
                f"Index: {result['index']}, Label: {result['label']}, "
                f"Distance: {result['distance']:.4f}"
            )

    except Exception as e:
        logger.error(f"Test error: {str(e)}")
