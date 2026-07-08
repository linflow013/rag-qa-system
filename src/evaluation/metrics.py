"""Custom metrics for evaluating RAG pipeline quality."""

import re
from typing import Dict, List, Set


def compute_token_overlap(text1: str, text2: str) -> float:
    """Simple token overlap between two texts."""
    tokens1 = set(re.findall(r"\w+", text1.lower()))
    tokens2 = set(re.findall(r"\w+", text2.lower()))
    if not tokens1 or not tokens2:
        return 0.0
    intersection = tokens1 & tokens2
    return len(intersection) / min(len(tokens1), len(tokens2))


def compute_faithfulness(
    answer: str, retrieved_chunks: List[str]
) -> float:
    """
    Simple faithfulness score: what fraction of answer claims appear in context.
    A claim is approximated as a sentence.
    """
    if not answer or not retrieved_chunks:
        return 0.0

    # Split answer into sentences
    answer_sentences = [
        s.strip() for s in re.split(r"[.。!！?\n]", answer) if len(s.strip()) > 10
    ]

    if not answer_sentences:
        return 0.0

    combined_context = " ".join(retrieved_chunks).lower()
    supported = 0

    for sentence in answer_sentences:
        # Check if the sentence's key tokens appear in the context
        key_tokens = set(re.findall(r"\w+", sentence.lower()))
        if not key_tokens:
            continue
        context_tokens = set(re.findall(r"\w+", combined_context))
        overlap = len(key_tokens & context_tokens) / max(len(key_tokens), 1)
        if overlap > 0.3:  # if >30% of key tokens found in context
            supported += 1

    return supported / max(len(answer_sentences), 1)


def compute_context_precision(
    question: str, retrieved_chunks: List[str], relevant_keywords: Set[str]
) -> float:
    """
    Context Precision: fraction of retrieved chunks that contain relevant keywords.
    """
    if not retrieved_chunks or not relevant_keywords:
        return 0.0

    relevant_count = 0
    for chunk in retrieved_chunks[:10]:  # check top 10
        chunk_lower = chunk.lower()
        if any(kw.lower() in chunk_lower for kw in relevant_keywords):
            relevant_count += 1

    return relevant_count / max(len(retrieved_chunks[:10]), 1)


def compute_answer_correctness(
    answer: str, ground_truth: str
) -> float:
    """
    Simple correctness: token overlap between answer and ground truth.
    This is a proxy; production should use LLM-as-judge or human evaluation.
    """
    return compute_token_overlap(answer, ground_truth)
