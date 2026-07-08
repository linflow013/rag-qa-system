"""Structured JSON-line logging for RAG observability."""

import json
import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, Optional

from config import config


class RAGLogger:
    """Writes structured JSON-line logs for each RAG query."""

    def __init__(self, log_dir: Optional[str] = None):
        self.log_dir = log_dir or config.logs_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self._logger = logging.getLogger("rag.structured")

    def _log_file(self) -> str:
        date_str = datetime.now().strftime("%Y-%m-%d")
        return os.path.join(self.log_dir, f"rag_{date_str}.jsonl")

    def log_query(self, entry: Dict[str, Any]):
        """Write a structured log entry as a JSON line."""
        entry.setdefault("timestamp", datetime.now().isoformat())
        entry["type"] = "query"

        # Redact PII before logging
        from src.generation.guard import redact_pii
        if "question" in entry:
            entry["question"] = redact_pii(str(entry["question"]))
        if "answer" in entry:
            entry["answer"] = redact_pii(str(entry["answer"]))

        try:
            with open(self._log_file(), "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            self._logger.error("Failed to write log: %s", e)

    def log_retrieval(
        self,
        session_id: str,
        question: str,
        num_results: int,
        top_score: float,
        latency_ms: float,
        retrieval_method: str = "hybrid",
    ):
        """Log retrieval step details."""
        self.log_query({
            "type": "retrieval",
            "session_id": session_id,
            "question": question,
            "num_results": num_results,
            "top_score": round(top_score, 4),
            "latency_ms": round(latency_ms, 2),
            "retrieval_method": retrieval_method,
        })

    def log_generation(
        self,
        session_id: str,
        question: str,
        answer: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        model_name: str,
        num_sources: int,
        turn_number: int,
    ):
        """Log generation step details."""
        self.log_query({
            "type": "generation",
            "session_id": session_id,
            "question": question,
            "answer": answer[:500],  # truncate for log readability
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": round(latency_ms, 2),
            "model_name": model_name,
            "num_sources": num_sources,
            "turn_number": turn_number,
        })

    def log_error(
        self,
        session_id: str,
        question: str,
        error_type: str,
        error_message: str,
    ):
        """Log errors and refusals."""
        self.log_query({
            "type": "error",
            "session_id": session_id,
            "question": question,
            "error_type": error_type,
            "error_message": error_message,
        })

    def log_full(
        self,
        session_id: str,
        question: str,
        answer: str,
        retrieval_latency_ms: float,
        generation_latency_ms: float,
        total_latency_ms: float,
        num_chunks: int,
        top_score: float,
        input_tokens: int,
        output_tokens: int,
        model_name: str,
        turn_number: int,
        citations: list,
    ):
        """Write a comprehensive log entry for the full RAG pipeline."""
        self.log_query({
            "type": "full",
            "session_id": session_id,
            "question": question,
            "answer": answer[:500],
            "retrieval_latency_ms": round(retrieval_latency_ms, 2),
            "generation_latency_ms": round(generation_latency_ms, 2),
            "total_latency_ms": round(total_latency_ms, 2),
            "num_chunks_retrieved": num_chunks,
            "top_score": round(top_score, 4),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "model_name": model_name,
            "turn_number": turn_number,
            "citations": citations,
        })


# Singleton
rag_logger = RAGLogger()
