"""FastAPI dependency providers for shared services."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import Depends, Request

from app.config import Settings, get_settings
from app.services.embeddings import EmbeddingService, get_embedding_service
from app.services.ingest import IngestService
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)


def _ensure_runtime_services(request: Request, settings: Settings) -> None:
    """Create embedding + FAISS singletons once per process/app."""
    if getattr(request.app.state, "vector_store", None) is not None:
        return

    embeddings = get_embedding_service(settings.embedding_model)
    dimension = embeddings.dimension
    store = VectorStore(settings.data_dir, dimension=dimension)
    request.app.state.embedding_service = embeddings
    request.app.state.vector_store = store
    logger.info("Initialized vector store at %s (dim=%s)", settings.data_dir, dimension)


def get_embedding_dep(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
) -> EmbeddingService:
    _ensure_runtime_services(request, settings)
    return request.app.state.embedding_service


def get_vector_store(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
) -> VectorStore:
    _ensure_runtime_services(request, settings)
    return request.app.state.vector_store


def get_ingest_service(
    settings: Annotated[Settings, Depends(get_settings)],
    embeddings: Annotated[EmbeddingService, Depends(get_embedding_dep)],
    store: Annotated[VectorStore, Depends(get_vector_store)],
) -> IngestService:
    return IngestService(settings=settings, embeddings=embeddings, store=store)
