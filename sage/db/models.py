"""SQLAlchemy ORM models: Document, Chunk, QueryLog, QueryCache — plus
Conversation/Message, which the reference (stateless, single-shot Q&A)
project has no equivalent of. Sage's UI needs a sidebar of past
conversations and resumable multi-turn sessions, so those two tables exist
purely to support that: linear turn history only, no branching or editing.
"""

from datetime import UTC, datetime

from sqlalchemy import JSON, Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True)
    filename = Column(String, nullable=False)
    title = Column(String, nullable=True)
    company = Column(String, nullable=True)
    fiscal_year = Column(String, nullable=True)
    doc_type = Column(String, nullable=True)
    source_path = Column(String, nullable=False)
    page_count = Column(Integer, nullable=False, default=0)
    embedding_model = Column(String, nullable=True)
    ingested_at = Column(DateTime, default=_utcnow)
    status = Column(String, default="pending")  # pending|processing|ready|failed
    # sha256 of the raw PDF bytes -- lets sage/ingest/pipeline.py detect an
    # identical file being re-ingested and skip creating a duplicate
    # Document/Chunk set. Nullable (rows ingested before this column existed
    # have no checksum backfilled -- see sage/db/database.py's init_db() for
    # the auto-migration that adds this column to a pre-existing db/sage.db,
    # and the ingest pipeline's module docstring for the dedup semantics of
    # a NULL checksum) but unique when set: SQL treats multiple NULLs as
    # distinct (never colliding with each other), so this doesn't affect
    # legacy rows, but it does close the race ingest_pdf's own
    # check-then-insert dedup can't fully close on its own -- two truly
    # concurrent uploads of the same new file both passing the "not found
    # yet" pre-check before either commits. The second commit fails with an
    # IntegrityError instead of silently creating a duplicate Document.
    checksum = Column(String, nullable=True, unique=True, index=True)

    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey("documents.id"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    page_number = Column(Integer, nullable=True)
    section_title = Column(String, nullable=True)
    text = Column(Text, nullable=False)
    char_start = Column(Integer, nullable=True)
    char_end = Column(Integer, nullable=True)
    token_count = Column(Integer, nullable=True)
    embedding_id = Column(String, nullable=True)  # id used in the Chroma collection
    created_at = Column(DateTime, default=_utcnow)

    document = relationship("Document", back_populates="chunks")


class QueryLog(Base):
    __tablename__ = "query_logs"

    id = Column(Integer, primary_key=True)
    session_id = Column(String, nullable=True)
    query_text = Column(Text, nullable=False)
    model_name = Column(String, nullable=True)
    embedding_model = Column(String, nullable=True)
    companies = Column(JSON, nullable=True)  # list[str] filter used for this query
    retrieved_chunk_ids = Column(JSON, nullable=True)
    top_k = Column(Integer, nullable=True)
    retrieval_latency_ms = Column(Float, nullable=True)
    generation_latency_ms = Column(Float, nullable=True)
    total_latency_ms = Column(Float, nullable=True)
    prompt_tokens = Column(Integer, nullable=True)
    completion_tokens = Column(Integer, nullable=True)
    total_tokens = Column(Integer, nullable=True)
    cache_hit = Column(Boolean, default=False)  # exact or semantic; see generation/cache.py
    cost_usd = Column(Float, default=0.0)  # estimated via generation/cost.py
    answer_text = Column(Text, nullable=True)
    error = Column(Text, nullable=True)
    created_at = Column(DateTime, default=_utcnow)


class QueryCache(Base):
    """Exact-match cache, keyed on hash(query + metadata filters + model).

    A row holds everything needed to reconstruct an AnswerResult without
    re-running retrieval or generation.
    """

    __tablename__ = "query_cache"

    id = Column(Integer, primary_key=True)
    cache_key = Column(String, unique=True, nullable=False, index=True)
    query_text = Column(Text, nullable=False)
    model_name = Column(String, nullable=True)
    answer_text = Column(Text, nullable=False)
    citations_json = Column(JSON, nullable=False)
    retrieved_chunk_ids = Column(JSON, nullable=True)
    prompt_tokens = Column(Integer, nullable=True)
    completion_tokens = Column(Integer, nullable=True)
    total_tokens = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=_utcnow)


class Conversation(Base):
    """A resumable multi-turn session, shown in the UI's conversation sidebar."""

    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True)
    title = Column(String, nullable=True)
    # Opaque, unguessable per-session identifier (secrets.token_urlsafe(32),
    # see sage/db/conversations.py:create_conversation) -- the only thing
    # that scopes a conversation to the visitor who created it. Every
    # conversation has one; there is no user-account concept.
    session_token = Column(String, nullable=False, index=True)
    created_at = Column(DateTime, default=_utcnow)

    messages = relationship(
        "Message",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.ordered",
    )


class Message(Base):
    """One turn in a Conversation. Linear history only -- no branching or
    editing of prior turns, by design (see module docstring)."""

    __tablename__ = "messages"

    id = Column(Integer, primary_key=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=False)
    role = Column(String, nullable=False)  # "user" | "assistant"
    content = Column(Text, nullable=False)
    citations = Column(JSON, nullable=True)  # list[dict], only populated for assistant turns
    ordered = Column(Integer, nullable=False)  # 0-indexed position within the conversation
    created_at = Column(DateTime, default=_utcnow)

    conversation = relationship("Conversation", back_populates="messages")
