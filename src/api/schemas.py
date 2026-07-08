"""Pydantic request/response schemas for the RAG QA API."""

from typing import List, Optional
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="Natural language question")
    session_id: Optional[str] = Field(None, description="Session ID for multi-turn conversation")
    top_k: int = Field(5, ge=1, le=20, description="Number of documents to retrieve")


class CitationResponse(BaseModel):
    source_file: str
    page_number: int
    excerpt: str = ""


class AskResponse(BaseModel):
    answer: str
    citations: List[CitationResponse] = []
    session_id: str
    latency_ms: float
    token_usage: dict = {}
    turn_number: int = 1


class HealthResponse(BaseModel):
    status: str
    version: str = "0.1.0"
    index_size: int = 0


class ErrorResponse(BaseModel):
    error: str
    detail: str = ""
    session_id: Optional[str] = None
