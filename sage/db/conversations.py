"""CRUD helpers for Conversation/Message — resumable multi-turn sessions.

Kept deliberately thin (linear turn history, no branching/editing) since the
only consumer right now is the CLI's `ask --conversation` flow; a future API
layer can import these same functions directly.
"""

import secrets
from dataclasses import dataclass

from sage.db.database import get_session
from sage.db.models import Conversation, Message


@dataclass
class HistoryTurn:
    role: str
    content: str


def create_conversation(
    title: str | None = None, session_token: str | None = None
) -> tuple[int, str]:
    """Create a conversation and return (id, session_token).

    `session_token` lets a caller that already holds one (a browser
    continuing its session with a second conversation) tag the new
    conversation with it, so it shows up alongside the caller's earlier
    conversations; omit it to mint a fresh one -- the CLI, which has no
    session concept, always takes this path.
    """
    session = get_session()
    try:
        token = session_token or secrets.token_urlsafe(32)
        conversation = Conversation(title=title, session_token=token)
        session.add(conversation)
        session.commit()
        session.refresh(conversation)
        return conversation.id, conversation.session_token
    finally:
        session.close()


def append_message(
    conversation_id: int,
    role: str,
    content: str,
    citations: list[dict] | None = None,
) -> int:
    """Append a turn, auto-assigning the next `ordered` position."""
    session = get_session()
    try:
        last = (
            session.query(Message)
            .filter(Message.conversation_id == conversation_id)
            .order_by(Message.ordered.desc())
            .first()
        )
        next_order = (last.ordered + 1) if last is not None else 0
        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            citations=citations or [],
            ordered=next_order,
        )
        session.add(message)
        session.commit()
        session.refresh(message)
        return message.id
    finally:
        session.close()


def get_history(conversation_id: int) -> list[HistoryTurn]:
    """Full linear turn history for a conversation, oldest first."""
    session = get_session()
    try:
        rows = (
            session.query(Message)
            .filter(Message.conversation_id == conversation_id)
            .order_by(Message.ordered)
            .all()
        )
        return [HistoryTurn(role=r.role, content=r.content) for r in rows]
    finally:
        session.close()


def list_conversations(session_token: str | None = None) -> list[Conversation]:
    """List conversations, newest first.

    `session_token=None` (the CLI's usage, which has no session concept)
    returns every conversation, unscoped. Any other value -- including ""
    for a caller with no session yet -- filters to conversations tagged
    with exactly that token.
    """
    session = get_session()
    try:
        query = session.query(Conversation)
        if session_token is not None:
            query = query.filter(Conversation.session_token == session_token)
        rows = query.order_by(Conversation.created_at.desc()).all()
        session.expunge_all()
        return rows
    finally:
        session.close()


def get_conversation(conversation_id: int, session_token: str | None = None) -> Conversation | None:
    """Fetch a conversation with its messages, or None if it doesn't exist.

    `session_token=None` (internal/CLI use) skips ownership checking. Any
    other value scopes the lookup to a conversation tagged with exactly
    that token -- a mismatch is indistinguishable from "doesn't exist".
    """
    session = get_session()
    try:
        query = session.query(Conversation).filter(Conversation.id == conversation_id)
        if session_token is not None:
            query = query.filter(Conversation.session_token == session_token)
        row = query.first()
        if row is not None:
            session.refresh(row, attribute_names=["messages"])
            session.expunge(row)
        return row
    finally:
        session.close()


def conversation_belongs_to_session(conversation_id: int, session_token: str) -> bool:
    """Whether `conversation_id` exists and is tagged with `session_token`."""
    session = get_session()
    try:
        return (
            session.query(Conversation.id)
            .filter(
                Conversation.id == conversation_id,
                Conversation.session_token == session_token,
            )
            .first()
            is not None
        )
    finally:
        session.close()
