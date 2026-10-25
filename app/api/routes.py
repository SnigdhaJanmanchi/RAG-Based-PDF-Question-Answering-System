"""HTTP routes for the AI PDF Assistant API."""

from __future__ import annotations

import logging
import time
from typing import Annotated

from fastapi import APIRouter, Depends, File, UploadFile

from app import __version__
from app.config import get_settings
from app.dependencies import get_ingest_service, get_rag_service, get_vector_store
from app.exceptions import AppError
from app.models.schemas import (
    AskDocumentRef,
    AskRequest,
    AskResponse,
    AskSource,
    DocumentInfo,
    DocumentListResponse,
    HealthResponse,
    SearchHitResponse,
    SearchRequest,
    SearchResponse,
    UploadResponse,
)
from app.services.ingest import IngestService
from app.services.rag import RAGService
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["health"])
def health() -> HealthResponse:
    """Return API liveness information."""
    settings = get_settings()
    logger.debug("Health check requested (env=%s)", settings.app_env)
    return HealthResponse(
        status="ok",
        app=settings.app_name,
        version=__version__,
    )


async def _read_and_ingest(
    file: UploadFile,
    ingest: IngestService,
) -> UploadResponse:
    started = time.perf_counter()
    data = await file.read()
    try:
        result = ingest.ingest_pdf(
            data,
            filename=file.filename or "upload.pdf",
            content_type=file.content_type,
        )
    except AppError:
        logger.info(
            "upload_failed status=error bytes=%s elapsed_ms=%.1f",
            len(data),
            (time.perf_counter() - started) * 1000,
        )
        raise
    except ValueError as exc:
        logger.info(
            "upload_failed status=400 bytes=%s elapsed_ms=%.1f",
            len(data),
            (time.perf_counter() - started) * 1000,
        )
        raise AppError(str(exc), status_code=400) from exc

    logger.info(
        "upload_ok duplicate=%s pages=%s chunks=%s bytes=%s elapsed_ms=%.1f",
        result.duplicate,
        result.page_count,
        result.chunk_count,
        len(data),
        (time.perf_counter() - started) * 1000,
    )
    return UploadResponse(
        document_id=result.document_id,
        filename=result.filename,
        page_count=result.page_count,
        chunk_count=result.chunk_count,
        duplicate=result.duplicate,
    )


@router.post(
    "/documents/upload",
    response_model=UploadResponse,
    tags=["documents"],
    summary="Upload and index a PDF",
)
async def upload_document(
    file: Annotated[UploadFile, File(description="PDF file to index")],
    ingest: Annotated[IngestService, Depends(get_ingest_service)],
) -> UploadResponse:
    """Accept a PDF, extract/chunk/embed, and index into FAISS."""
    return await _read_and_ingest(file, ingest)


@router.post(
    "/upload",
    response_model=UploadResponse,
    tags=["documents"],
    summary="Upload alias (same as /documents/upload)",
    include_in_schema=True,
)
async def upload_alias(
    file: Annotated[UploadFile, File(description="PDF file to index")],
    ingest: Annotated[IngestService, Depends(get_ingest_service)],
) -> UploadResponse:
    """Compatibility alias for ``POST /documents/upload``."""
    return await _read_and_ingest(file, ingest)


@router.get(
    "/documents",
    response_model=DocumentListResponse,
    tags=["documents"],
)
def list_documents(
    store: Annotated[VectorStore, Depends(get_vector_store)],
) -> DocumentListResponse:
    """List indexed documents."""
    docs = [
        DocumentInfo(
            document_id=d.doc_id,
            filename=d.filename,
            page_count=d.page_count,
            chunk_count=d.chunk_count,
        )
        for d in store.list_documents()
    ]
    return DocumentListResponse(documents=docs)


@router.post(
    "/search",
    response_model=SearchResponse,
    tags=["search"],
    summary="Semantic search over indexed chunks (no LLM)",
)
def search(
    body: SearchRequest,
    ingest: Annotated[IngestService, Depends(get_ingest_service)],
) -> SearchResponse:
    """Retrieve top-k chunks for a query. Does not call an LLM."""
    started = time.perf_counter()
    try:
        hits = ingest.search(
            body.query,
            top_k=body.top_k,
            doc_id=body.document_id,
        )
    except KeyError as exc:
        raise AppError(str(exc), status_code=404) from exc
    except ValueError as exc:
        raise AppError(str(exc), status_code=400) from exc

    logger.info(
        "search_ok hits=%s elapsed_ms=%.1f",
        len(hits),
        (time.perf_counter() - started) * 1000,
    )
    return SearchResponse(
        query=body.query,
        results=[
            SearchHitResponse(
                score=h.score,
                text=h.text,
                document_id=h.doc_id,
                chunk_id=h.chunk_id,
                chunk_index=h.chunk_index,
                page=h.page,
                filename=h.filename,
            )
            for h in hits
        ],
    )


@router.post(
    "/ask",
    response_model=AskResponse,
    tags=["ask"],
    summary="Answer a question using retrieved PDF context (OpenAI)",
)
def ask(
    body: AskRequest,
    rag: Annotated[RAGService, Depends(get_rag_service)],
) -> AskResponse:
    """Grounded RAG answer. Does not use external knowledge beyond retrieved chunks."""
    started = time.perf_counter()
    try:
        result = rag.ask(
            body.question,
            document_id=body.document_id,
            top_k=body.top_k,
        )
    except KeyError as exc:
        raise AppError(str(exc), status_code=404) from exc
    except ValueError as exc:
        raise AppError(str(exc), status_code=400) from exc
    except AppError:
        logger.info(
            "ask_failed elapsed_ms=%.1f",
            (time.perf_counter() - started) * 1000,
        )
        raise

    logger.info(
        "ask_ok status=%s sources=%s elapsed_ms=%.1f",
        result.status,
        len(result.sources),
        (time.perf_counter() - started) * 1000,
    )
    return AskResponse(
        question=result.question,
        answer=result.answer,
        status=result.status,
        sources=[
            AskSource(
                document_id=s.document_id,
                filename=s.filename,
                page=s.page,
                chunk_id=s.chunk_id,
                excerpt=s.excerpt,
                score=s.score,
            )
            for s in result.sources
        ],
        documents=[
            AskDocumentRef(document_id=d.document_id, filename=d.filename)
            for d in result.documents
        ],
    )
