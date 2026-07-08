"""Session management for the FastAPI layer."""

import uuid
from fastapi import Request

from src.conversation.memory import ConversationMemory

# Global session store
memory = ConversationMemory()


def get_or_create_session(request: Request) -> str:
    """Extract or create session_id from request headers."""
    session_id = request.headers.get("X-Session-ID", "")
    if session_id:
        return memory.get_or_create(session_id)
    new_id = str(uuid.uuid4())[:8]
    return memory.get_or_create(new_id)
