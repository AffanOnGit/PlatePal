"""
PlatePal - CLIP Embedding Extractor

Provides a unified interface for extracting fixed 512-dim text embeddings
from recipe titles and ingredient lists using OpenAI's CLIP model.
These embeddings serve as the conditioning vector for the DCGAN Generator.
"""

import torch
import numpy as np
from typing import List, Union


class CLIPEmbedder:
    """
    Wraps the CLIP model to produce fixed-size text embeddings
    for conditioning the image generator.
    """

    def __init__(self, model_name: str = "ViT-B/32", device: str = None):
        """
        Args:
            model_name: CLIP architecture variant.
            device: 'cuda' or 'cpu'. Auto-detected if None.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.model = None
        self.preprocess = None
        self._tokenize = None
        self._loaded = False

    def load(self):
        """Lazy-load the CLIP model (downloads on first use)."""
        if self._loaded:
            return

        try:
            import clip
            self.model, self.preprocess = clip.load(self.model_name, device=self.device)
            self._tokenize = clip.tokenize
            self.model.eval()
            self._loaded = True
            print(f"[CLIPEmbedder] Loaded {self.model_name} on {self.device}")
        except ImportError:
            print(
                "[CLIPEmbedder] WARNING: 'clip' package not installed. "
                "Install with: pip install git+https://github.com/openai/CLIP.git"
            )
            raise

    @torch.no_grad()
    def encode_text(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """
        Encode one or more text strings into CLIP embeddings.

        Args:
            texts: A single string or list of strings.

        Returns:
            Tensor of shape (N, 512) — L2-normalized embeddings.
        """
        self.load()

        if isinstance(texts, str):
            texts = [texts]

        # CLIP tokenizer truncates to 77 tokens automatically
        tokens = self._tokenize(texts).to(self.device)
        features = self.model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.float()

    @torch.no_grad()
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode batch of images into CLIP embeddings.

        Args:
            images: Tensor of shape (N, 3, 224, 224) preprocessed for CLIP.

        Returns:
            Tensor of shape (N, 512) — L2-normalized embeddings.
        """
        self.load()

        features = self.model.encode_image(images.to(self.device))
        features = features / features.norm(dim=-1, keepdim=True)
        return features.float()

    def embed_recipe(self, title: str, ingredients: str) -> torch.Tensor:
        """
        Build a conditioning embedding from recipe metadata.

        Concatenates title and ingredients into a descriptive prompt
        and encodes it with CLIP.

        Args:
            title: Recipe title string.
            ingredients: Comma-separated ingredient list.

        Returns:
            Tensor of shape (1, 512).
        """
        prompt = f"A photo of {title} made with {ingredients}"
        # Truncate if excessively long (CLIP handles 77 tokens)
        if len(prompt) > 300:
            prompt = prompt[:300]
        return self.encode_text(prompt)


class DummyCLIPEmbedder:
    """
    A fallback embedder that returns random 512-dim vectors.
    Use this when CLIP is not installed or for quick prototyping.
    """

    def __init__(self, embed_dim: int = 512, device: str = None):
        self.embed_dim = embed_dim
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print("[DummyCLIPEmbedder] Using random embeddings (CLIP not loaded)")

    def encode_text(self, texts: Union[str, List[str]]) -> torch.Tensor:
        if isinstance(texts, str):
            texts = [texts]
        return torch.randn(len(texts), self.embed_dim).to(self.device)

    def embed_recipe(self, title: str, ingredients: str) -> torch.Tensor:
        return torch.randn(1, self.embed_dim).to(self.device)


def get_embedder(use_clip: bool = True, **kwargs):
    """
    Factory function: returns a real CLIP embedder if available,
    otherwise falls back to DummyCLIPEmbedder.
    """
    if use_clip:
        try:
            embedder = CLIPEmbedder(**kwargs)
            embedder.load()
            return embedder
        except (ImportError, Exception) as e:
            print(f"[get_embedder] Falling back to dummy embedder: {e}")

    return DummyCLIPEmbedder(**kwargs)


if __name__ == "__main__":
    embedder = get_embedder(use_clip=True)
    emb = embedder.embed_recipe("Chicken Biryani", "chicken, rice, onion, yogurt, spices")
    print(f"Embedding shape: {emb.shape}")  # (1, 512)
    print(f"Embedding norm:  {emb.norm().item():.4f}")
