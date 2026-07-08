"""Conversation memory — stores and manages multi-turn dialogue history."""

import logging
import time
from typing import Dict, List, Optional

from src.generation.llm_client import Message

logger = logging.getLogger(__name__)


class ConversationMemory:
    """
    In-memory conversation buffer keyed by session_id.
    Tracks turn history and supports auto-summarization for long conversations.
    """

    def __init__(self, ttl_seconds: int = 1800, max_turns: int = 10):
        self._store: Dict[str, dict] = {}
        self.ttl_seconds = ttl_seconds
        self.max_turns = max_turns

    def _cleanup_expired(self):
        """Remove expired sessions."""
        now = time.time()
        expired = [
            sid for sid, data in self._store.items()
            if now - data.get("last_access", 0) > self.ttl_seconds
        ]
        for sid in expired:
            del self._store[sid]
        if expired:
            logger.info("Cleaned up %d expired sessions", len(expired))

    def get_or_create(self, session_id: str) -> str:
        """Get existing session or create a new one. Returns the session_id."""
        self._store.setdefault(session_id, {
            "messages": [],
            "created_at": time.time(),
            "last_access": time.time(),
            "turn_count": 0,
        })
        self._store[session_id]["last_access"] = time.time()
        return session_id

    def add_turn(self, session_id: str, user_msg: str, assistant_msg: str):
        """Record a conversation turn."""
        if session_id not in self._store:
            self.get_or_create(session_id)

        session = self._store[session_id]
        session["messages"].append(Message(role="user", content=user_msg))
        session["messages"].append(Message(role="assistant", content=assistant_msg))
        session["turn_count"] += 1
        session["last_access"] = time.time()

        # Truncate if too long
        if session["turn_count"] > self.max_turns:
            # Keep only recent turns
            excess = (session["turn_count"] - self.max_turns) * 2
            session["messages"] = session["messages"][excess:]
            session["turn_count"] = self.max_turns

    def get_history(self, session_id: str, max_turns: int = 5) -> List[Message]:
        """Get recent conversation history."""
        if session_id not in self._store:
            return []

        messages = self._store[session_id]["messages"]
        return messages[-(max_turns * 2):]

    def clear(self, session_id: str):
        """Clear a session's history."""
        if session_id in self._store:
            del self._store[session_id]
            logger.info("Cleared session %s", session_id)

    def stats(self) -> dict:
        """Return memory statistics."""
        return {
            "active_sessions": len(self._store),
            "total_turns": sum(s["turn_count"] for s in self._store.values()),
        }
