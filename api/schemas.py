"""Pydantic request/response models. Every top-level response carries
schema_version -- bump it on any breaking response-shape change."""

from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, Field, StringConstraints

from config import settings

SCHEMA_VERSION = 1


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=settings.MAX_QUERY_LENGTH)
    companies: (
        list[
            Annotated[
                str,
                StringConstraints(
                    strip_whitespace=True,
                    min_length=1,
                    max_length=settings.MAX_COMPANY_FILTER_LENGTH,
                ),
            ]
        ]
        | None
    ) = Field(default=None, max_length=settings.MAX_COMPARISON_COMPANIES)
    fiscal_year: str | None = None
    doc_type: str | None = None
    top_k: int = Field(
        default=settings.DEFAULT_TOP_K,
        ge=1,
        le=settings.RERANK_CANDIDATE_K,
        strict=True,
    )
    session_id: int | None = None  # conversation id to continue, if any


class CitationOut(BaseModel):
    n: int
    chunk_id: int
    text: str
    page_number: int | None
    company: str | None
    fiscal_year: str | None
    doc_type: str | None
    filename: str


class LatencyOut(BaseModel):
    retrieval_ms: float
    generation_ms: float
    total_ms: float


class TokensOut(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatResponse(BaseModel):
    schema_version: int = SCHEMA_VERSION
    answer: str
    citations: list[CitationOut]
    model: str
    latency_ms: LatencyOut
    tokens: TokensOut
    cache_hit: bool
    cost_usd: float
    session_id: int | None = None


class ConversationCreateRequest(BaseModel):
    title: str | None = None


class ConversationCreateResponse(BaseModel):
    schema_version: int = SCHEMA_VERSION
    conversation_id: int
    session_token: str


class ConversationSummaryOut(BaseModel):
    schema_version: int = SCHEMA_VERSION
    id: int
    title: str | None
    created_at: datetime


class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    citations: list[dict] | None
    created_at: datetime


class ConversationDetailResponse(BaseModel):
    schema_version: int = SCHEMA_VERSION
    id: int
    title: str | None
    created_at: datetime
    messages: list[MessageOut]


class DocumentOut(BaseModel):
    schema_version: int = SCHEMA_VERSION
    id: int
    filename: str
    title: str | None
    company: str | None
    fiscal_year: str | None
    doc_type: str | None
    page_count: int
    status: str
    ingested_at: datetime


class UploadResponse(BaseModel):
    schema_version: int = SCHEMA_VERSION
    document_id: int
    filename: str
    status: str
    page_count: int
    company: str | None
    fiscal_year: str | None
    doc_type: str | None
