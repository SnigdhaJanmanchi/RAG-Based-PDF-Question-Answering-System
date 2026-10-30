"""Tests for embedding singleton, dims, and normalization."""

from __future__ import annotations

import numpy as np
import pytest

from app.services.embeddings import get_embedding_service, reset_embedding_services


@pytest.fixture(scope="module")
def embedding_service():
    reset_embedding_services()
    service = get_embedding_service("sentence-transformers/all-MiniLM-L6-v2")
    # Force load once for the module.
    _ = service.dimension
    yield service


def test_singleton_same_instance(embedding_service) -> None:
    again = get_embedding_service("sentence-transformers/all-MiniLM-L6-v2")
    assert again is embedding_service


def test_embed_documents_normalized_and_consistent_dim(embedding_service) -> None:
    texts = ["hello world", "indigo phoenix at Lake Azure"]
    vectors = embedding_service.embed_documents(texts, batch_size=2)
    assert vectors.shape == (2, embedding_service.dimension)
    norms = np.linalg.norm(vectors, axis=1)
    np.testing.assert_allclose(norms, np.ones(2), atol=1e-5)


def test_embed_query_shape(embedding_service) -> None:
    vector = embedding_service.embed_query("Lake Azure")
    assert vector.shape == (embedding_service.dimension,)
    assert abs(float(np.linalg.norm(vector)) - 1.0) < 1e-5
