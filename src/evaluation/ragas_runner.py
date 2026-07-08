"""RAGAS evaluation runner and reporting."""

import json
import logging
import os
import time
from typing import List

from src.evaluation.test_cases import TEST_QA_PAIRS
from src.indexing.embedder import embedder
from src.indexing.vector_store import VectorStore
from src.indexing.bm25_index import BM25Index
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.sparse_retriever import SparseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.llm_client import get_llm_client
from src.generation.prompt_builder import build_prompt, build_system_prompt
from config import config

logger = logging.getLogger(__name__)

# Try importing RAGAS
try:
    from ragas.metrics import faithfulness as ragas_faithfulness
    from ragas.metrics import context_precision as ragas_context_precision
    HAS_RAGAS = True
except ImportError:
    HAS_RAGAS = False


def run_evaluation(
    vector_store: VectorStore,
    bm25_index: BM25Index,
    test_cases: List[dict] = None,
) -> dict:
    """Run full evaluation suite and return results."""
    if test_cases is None:
        test_cases = TEST_QA_PAIRS

    dense = DenseRetriever(vector_store)
    sparse = SparseRetriever(bm25_index)
    hybrid = HybridRetriever(dense, sparse, rrf_k=config.retrieval.rrf_k)
    llm = get_llm_client()

    results = []
    total_retrieval_latency = 0
    total_generation_latency = 0
    answered_out_of_scope = 0

    for i, case in enumerate(test_cases):
        logger.info("Evaluating [%d/%d]: %s", i + 1, len(test_cases), case["question"][:60])

        # Retrieval
        r_start = time.time()
        retrieval_results = hybrid.retrieve(case["question"], top_k=5)
        r_latency = (time.time() - r_start) * 1000
        total_retrieval_latency += r_latency

        retrieved_texts = [r.text for r in retrieval_results]
        retrieved_sources = [r.metadata.get("source_file", "") for r in retrieval_results]

        # Generation
        g_start = time.time()
        lang = case.get("language", "en")
        system_prompt = build_system_prompt(lang)

        if retrieval_results:
            prompt = build_prompt(case["question"], retrieval_results)
            gen_result = llm.generate(prompt=prompt, system_prompt=system_prompt)
        else:
            gen_result = type('obj', (object,), {
                'text': '[No relevant documents found]',
                'input_tokens': 0,
                'output_tokens': 0,
                'latency_ms': 0,
                'model_name': 'none',
            })()

        g_latency = gen_result.latency_ms
        total_generation_latency += g_latency

        answer = gen_result.text

        # Check if out-of-scope question was properly refused
        is_out_of_scope = case.get("relevant_doc") is None
        if is_out_of_scope:
            refusal_keywords = ["cannot", "could not", "无法", "没有找到", "not found"]
            properly_refused = any(kw in answer.lower() for kw in refusal_keywords)
            if properly_refused:
                answered_out_of_scope += 1

        # Simple faithfulness metric
        from src.evaluation.metrics import (
            compute_faithfulness,
            compute_answer_correctness,
        )
        faithfulness = compute_faithfulness(answer, retrieved_texts)
        correctness = compute_answer_correctness(answer, case["ground_truth"])

        results.append({
            "index": i + 1,
            "question": case["question"],
            "language": case.get("language", "en"),
            "is_out_of_scope": is_out_of_scope,
            "faithfulness": round(faithfulness, 4),
            "answer_correctness": round(correctness, 4),
            "num_retrieved": len(retrieval_results),
            "retrieval_latency_ms": round(r_latency, 2),
            "generation_latency_ms": round(g_latency, 2),
            "answer": answer[:300],
            "ground_truth": case["ground_truth"][:200],
        })

    # Aggregate metrics
    n = len(results)
    avg_faithfulness = sum(r["faithfulness"] for r in results) / max(n, 1)
    avg_correctness = sum(r["answer_correctness"] for r in results) / max(n, 1)
    avg_r_latency = total_retrieval_latency / max(n, 1)
    avg_g_latency = total_generation_latency / max(n, 1)
    out_of_scope_count = sum(1 for r in results if r["is_out_of_scope"])
    out_of_scope_correct = answered_out_of_scope

    summary = {
        "total_test_cases": n,
        "avg_faithfulness": round(avg_faithfulness, 4),
        "faithfulness_target": 0.85,
        "faithfulness_pass": avg_faithfulness >= 0.85,
        "avg_answer_correctness": round(avg_correctness, 4),
        "accuracy_target": 0.80,
        "accuracy_pass": avg_correctness >= 0.80,
        "avg_retrieval_latency_ms": round(avg_r_latency, 2),
        "avg_generation_latency_ms": round(avg_g_latency, 2),
        "avg_total_latency_ms": round(avg_r_latency + avg_g_latency, 2),
        "out_of_scope_questions": out_of_scope_count,
        "out_of_scope_correctly_refused": out_of_scope_correct,
        "out_of_scope_refusal_rate": (
            out_of_scope_correct / out_of_scope_count if out_of_scope_count > 0 else 1.0
        ),
    }

    return {
        "summary": summary,
        "details": results,
    }
