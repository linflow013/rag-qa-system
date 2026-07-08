"""Guard module — prompt injection defense, PII handling, refusal logic."""

import logging
import re
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

# ─── Prompt injection patterns ──────────────────────────────────────

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|above|prior)\s+instructions?",
    r"forget\s+(all\s+)?(previous|your)\s+(instructions|prompt)",
    r"you\s+are\s+now\s+(a\s+)?(different|new)\s+(ai|assistant|role)",
    r"system\s*:\s*you\s+are",
    r"\[system\s*prompt\]",
    r"<\|system\|>",
    r"pretend\s+you\s+are",
    r"act\s+as\s+(if\s+)?(you\s+are|a)\s+(?!an?\s+(enterprise|QA|assistant))",
]

INJECTION_REGEX = re.compile(
    "|".join(f"({p})" for p in INJECTION_PATTERNS),
    re.IGNORECASE,
)

# ─── PII patterns ───────────────────────────────────────────────────

PII_PATTERNS = [
    (re.compile(r"\d{3}-\d{4}-\d{4}"), "[REDACTED-PHONE]"),       # Phone
    (re.compile(r"\d{17}[\dXx]"), "[REDACTED-ID]"),               # CN ID number
    (re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"), "[REDACTED-EMAIL]"),
    (re.compile(r"\b(?:\d[ -]*?){13,16}\b"), "[REDACTED-CARD]"),  # Credit card
]


def detect_injection(question: str) -> bool:
    """Check if the question contains prompt injection patterns."""
    if not question:
        return False
    match = INJECTION_REGEX.search(question)
    if match:
        logger.warning("Prompt injection detected in question: pattern matched")
        return True
    return False


def redact_pii(text: str) -> str:
    """Replace PII patterns with redacted placeholders."""
    for pattern, replacement in PII_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def check_retrieval_quality(
    results: List,
    score_threshold: float = 0.1,
) -> Tuple[bool, str]:
    """
    Check if retrieval results are sufficient for answering.
    Returns (is_ok, refusal_message).
    """
    if not results:
        return False, (
            "I could not find any relevant information in the knowledge base "
            "to answer your question. Please try rephrasing or ask a different "
            "question about AcmeTech's policies, systems, or documentation."
        )

    max_score = max(r.score for r in results) if results else 0

    if max_score < score_threshold:
        return False, (
            "I found some documents but they have very low relevance to your "
            "question (best match score: {:.2f}). I cannot provide a reliable "
            "answer based on the available information. Please try rephrasing "
            "your question."
        ).format(max_score)

    return True, ""


def detect_language_from_results(results: List) -> str:
    """Determine the answer language based on retrieved documents."""
    cn_count = sum(1 for r in results if r.metadata.get("language") == "cn")
    total = len(results)
    if total > 0 and cn_count / total >= 0.5:
        return "cn"
    return "en"
