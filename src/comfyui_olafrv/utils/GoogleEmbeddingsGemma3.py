import os
import torch
from typing import Optional
from sentence_transformers import SentenceTransformer


class GoogleEmbeddingsGemma3:
    """Utility class to handle loading and using the
    Google Embedding Gemma 3 model for text embeddings.
    https://ai.google.dev/gemma/docs/embeddinggemma/inference-embeddinggemma-with-sentence-transformers
    """

    def __init__(self, model_path: str):
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.model: Optional[SentenceTransformer] = None
        try:
            self.model = self._load_local_model(model_path)
            print(f"Device: {self.device}")
            print(f"Model: {self.model}")
            parameters = sum(p.numel() for p in self.model.parameters())
            print("Model parameters:", parameters)
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {str(e)}")

    def _load_local_model(self, local_model_path: str) -> SentenceTransformer:
        """
        Load the SentenceTransformer model from a local path.
        """
        if not os.path.exists(local_model_path):
            raise FileNotFoundError(f"Model path does not exist: {local_model_path}")

        try:
            self.model = SentenceTransformer(local_model_path).to(self.device)
            return self.model
        except Exception as e:
            raise Exception(f"Failed to load SentenceTransformer model: {str(e)}")

    def calculate_embeddings(self, text: str, convert_to_tensor: bool) -> torch.Tensor:
        """
        Calculate a pooled 1x768 embedding for the given text
        using the SentenceTransformer model.

        Parameters
        ----------
        text : str
            Input text to embed.
        pooling : str
            Pooling strategy: "mean", "cls", or "max".
        """
        if self.model is None:
            raise RuntimeError("Please load the model before encoding.")

        try:
            # strip line breaks and spaces
            cleaned_text = text.strip()
            print(f"Cleaned Text: '{cleaned_text}'")
            if not cleaned_text:
                raise ValueError("Input text is empty")

            # Force tensor output for consistency
            embeddings = self.model.encode(cleaned_text, convert_to_tensor=convert_to_tensor)
            print(f"(Pooled) Embeddings (Shape): {embeddings.shape}")
            return embeddings

        except Exception as e:
            raise RuntimeError(f"Failed to encode text: {str(e)}")
