"""Sentence Transformers embeddings with a process-wide singleton model."""

from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_instances: dict[str, "EmbeddingService"] = {}


class EmbeddingService:
    """Wrapper around a Sentence Transformers model (loaded once per model name)."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model: Any | None = None
        self._dimension: int | None = None

    def _ensure_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model '%s' (one-time)", self.model_name)
            self._model = SentenceTransformer(self.model_name)
            # Probe dimension with a tiny encode.
            probe = self._model.encode(
                ["dimension probe"],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            self._dimension = int(np.asarray(probe).shape[-1])
            logger.info(
                "Embedding model ready (dim=%s)",
                self._dimension,
            )
        return self._model

    @property
    def dimension(self) -> int:
        self._ensure_model()
        assert self._dimension is not None
        return self._dimension

    def embed_documents(
        self, texts: list[str], *, batch_size: int = 32
    ) -> np.ndarray:
        """Embed a batch of document chunks. Returns (n, dim) float32, L2-normalized."""
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        model = self._ensure_model()
        vectors = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        arr = np.asarray(vectors, dtype=np.float32)
        if arr.ndim != 2:
            raise RuntimeError(f"Unexpected embedding shape: {arr.shape}")
        return arr

    def embed_query(self, text: str) -> np.ndarray:
        """Embed a single query string. Returns (dim,) float32, L2-normalized."""
        vectors = self.embed_documents([text], batch_size=1)
        return vectors[0]


def get_embedding_service(model_name: str) -> EmbeddingService:
    """Return a cached EmbeddingService for the given model name (singleton)."""
    with _lock:
        service = _instances.get(model_name)
        if service is None:
            service = EmbeddingService(model_name)
            _instances[model_name] = service
        return service


def reset_embedding_services() -> None:
    """Clear cached models (for tests)."""
    with _lock:
        _instances.clear()
