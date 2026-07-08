"""Embedding model wrapper — BAAI/bge-m3 for dense + sparse vectors.

For users in China who cannot access huggingface.co directly, set:
    HF_ENDPOINT=https://hf-mirror.com
Alternatively, use a local model path or a lighter model like:
    sentence-transformers/all-MiniLM-L6-v2 (~80MB, English only)
"""

import logging
import os
from typing import Dict, List, Optional

import numpy as np

from config import config

logger = logging.getLogger(__name__)

# Fallback model if the primary model is unavailable.
# all-MiniLM-L6-v2 is ~80MB (vs 2.2GB for bge-m3) and works well for English.
FALLBACK_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    """Encodes text into dense vectors using a sentence-transformer model."""

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or config.embedding.model_name
        self._model = None
        self.loaded = False
        self._dense_dim = config.embedding.dense_dim

    def load(self):
        """Load the embedding model into memory. Called once at startup."""
        if self.loaded:
            return

        models_to_try = [self.model_name]
        if self.model_name != FALLBACK_MODEL:
            models_to_try.append(FALLBACK_MODEL)

        last_error = None
        for model_name in models_to_try:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info("Loading embedding model: %s ...", model_name)
                self._model = SentenceTransformer(
                    model_name, device=config.embedding.device
                )
                self.model_name = model_name
                self._dense_dim = self._model.get_sentence_embedding_dimension()
                self.loaded = True
                logger.info("Embedding model loaded successfully (dim=%d).", self._dense_dim)
                return
            except Exception as e:
                last_error = e
                if model_name != models_to_try[-1]:
                    logger.warning(
                        "Failed to load %s, trying fallback %s. "
                        "Tip: set HF_ENDPOINT=https://hf-mirror.com for Chinese users.",
                        model_name, models_to_try[-1],
                    )

        logger.error("Failed to load any embedding model. Last error: %s", last_error)
        raise last_error

    def encode_dense(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        """
        Encode texts to dense vectors.
        For queries, prepend the bge-m3 instruction prefix.
        """
        if not self.loaded:
            self.load()

        if is_query:
            texts = [f"{config.embedding.query_instruction}{t}" for t in texts]

        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 50,
            batch_size=32,
        )
        return embeddings

    def encode_sparse(self, texts: List[str]) -> List[Dict[int, float]]:
        """
        Encode texts to sparse lexical vectors.
        When bge-m3 is loaded, uses its native sparse output.
        Otherwise returns empty dicts (fallback: BM25 handles sparse retrieval).
        """
        if not self.loaded:
            self.load()

        if "bge-m3" not in self.model_name.lower():
            # Fallback model doesn't support sparse encoding — BM25 handles it
            return [{} for _ in texts]

        try:
            sparse_embeddings = self._model.encode(
                texts,
                normalize_embeddings=False,
                show_progress_bar=len(texts) > 50,
                batch_size=32,
                output_value="sparse",
            )
            return sparse_embeddings
        except Exception:
            return [{} for _ in texts]

    def encode_single(self, text: str, is_query: bool = False) -> np.ndarray:
        """Convenience: encode a single text."""
        return self.encode_dense([text], is_query=is_query)[0]


# Singleton
embedder = Embedder()
