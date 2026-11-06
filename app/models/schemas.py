"""API request and response models."""

from typing import Literal

from pydantic import BaseModel, Field

RetrievalType = Literal["overview", "similarity"]


class HealthResponse(BaseModel):
    """Liveness / readiness payload for GET /health."""

    status: str = Field(description="Service health status")
    app: str = Field(description="Application name")
    version: str = Field(description="Application version")


class UploadResponse(BaseModel):
    """Result of indexing an uploaded PDF."""

    document_id: str
    filename: str
    page_count: int
    chunk_count: int
    duplicate: bool = Field(
        default=False,
        description="True when identical content was already indexed",
    )


class DocumentInfo(BaseModel):
    """Summary of an indexed document."""

    document_id: str
    filename: str
    page_count: int
    chunk_count: int


class DocumentListResponse(BaseModel):
    documents: list[DocumentInfo]


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, description="Natural-language search query")
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=50,
        description="Override default TOP_K for this request",
    )
    document_id: str | None = Field(
        default=None,
        description="Optional: restrict search to one document",
    )


class SearchHitResponse(BaseModel):
    score: float
    text: str
    document_id: str
    chunk_id: str
    chunk_index: int
    page: int
    filename: str
    retrieval: RetrievalType = Field(
        default="similarity",
        description="'similarity' for FAISS cosine hits; 'overview' for document-ordered chunks",
    )


class SearchResponse(BaseModel):
    query: str
    results: list[SearchHitResponse]


class AskRequest(BaseModel):
    question: str = Field(min_length=1, description="Natural-language question")
    document_id: str | None = Field(
        default=None,
        description="Optional: answer using only one indexed document",
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=50,
        description="Override default TOP_K for retrieval",
    )


class AskSource(BaseModel):
    document_id: str
    filename: str
    page: int
    chunk_id: str
    excerpt: str
    score: float = Field(
        description="FAISS cosine similarity when retrieval='similarity'; "
        "not a cosine score when retrieval='overview' (placeholder 0.0)"
    )
    retrieval: RetrievalType = Field(
        default="similarity",
        description="'similarity' for semantic FAISS hits; 'overview' for "
        "document-ordered leading chunks (not a cosine of 1.0)",
    )


class AskDocumentRef(BaseModel):
    document_id: str
    filename: str


class AskResponse(BaseModel):
    question: str
    answer: str
    status: str = Field(
        description="'ok' when an LLM answer was produced; "
        "'insufficient_evidence' when retrieval was empty/weak"
    )
    sources: list[AskSource]
    documents: list[AskDocumentRef]
