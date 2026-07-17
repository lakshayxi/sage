"""POST /conversations, GET /conversations, GET /conversations/{id}.

All three are scoped by an opaque `X-Session-Token` header the frontend
generates via the server on first conversation creation and persists in
localStorage -- see sage/db/conversations.py's module docstring and
create_conversation for why this exists (there's no user-account concept,
just per-browser isolation so one visitor can't read another's history).
A missing/wrong token gets an empty list (GET /conversations) or a 404
(GET /conversations/{id}, never a 403, so it doesn't confirm the id exists).
"""

from fastapi import APIRouter, Header, HTTPException

from api.schemas import (
    ConversationCreateRequest,
    ConversationCreateResponse,
    ConversationDetailResponse,
    ConversationSummaryOut,
    MessageOut,
)
from sage.db.conversations import create_conversation, get_conversation, list_conversations

router = APIRouter()


@router.post("/conversations", response_model=ConversationCreateResponse)
def create(
    body: ConversationCreateRequest,
    x_session_token: str = Header(default="", alias="X-Session-Token"),
) -> ConversationCreateResponse:
    # Reuse the caller's existing token (if it sent one) so a second
    # conversation from the same browser lands in the same session instead
    # of minting a new, disconnected one.
    conversation_id, session_token = create_conversation(
        title=body.title, session_token=x_session_token or None
    )
    return ConversationCreateResponse(conversation_id=conversation_id, session_token=session_token)


@router.get("/conversations", response_model=list[ConversationSummaryOut])
def list_all(
    x_session_token: str = Header(default="", alias="X-Session-Token"),
) -> list[ConversationSummaryOut]:
    return [
        ConversationSummaryOut(id=c.id, title=c.title, created_at=c.created_at)
        for c in list_conversations(session_token=x_session_token)
    ]


@router.get("/conversations/{conversation_id}", response_model=ConversationDetailResponse)
def get_one(
    conversation_id: int,
    x_session_token: str = Header(default="", alias="X-Session-Token"),
) -> ConversationDetailResponse:
    conversation = get_conversation(conversation_id, session_token=x_session_token)
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return ConversationDetailResponse(
        id=conversation.id,
        title=conversation.title,
        created_at=conversation.created_at,
        messages=[
            MessageOut(
                id=m.id,
                role=m.role,
                content=m.content,
                citations=m.citations,
                created_at=m.created_at,
            )
            for m in conversation.messages
        ],
    )
