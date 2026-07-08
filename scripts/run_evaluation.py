#!/usr/bin/env python
"""
Run the full RAG evaluation suite and print results.

Usage:
    python scripts/run_evaluation.py
    python scripts/run_evaluation.py --output eval_results.json
"""
import argparse
import json
import logging
import sys

sys.path.insert(0, ".")

from config import config
from src.indexing.embedder import embedder
from src.indexing.vector_store import VectorStore
from src.indexing.bm25_index import BM25Index
from src.ingestion.chunker import Document
from src.evaluation.ragas_runner import run_evaluation

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def main():
    parser = argparse.ArgumentParser(description="Run RAG evaluation suite")
    parser.add_argument("--output", default="logs/eval_results.json", help="Output file path")
    args = parser.parse_args()

    print("=" * 60)
    print("  RAG QA System - Evaluation Suite")
    print("=" * 60)
    print()

    # Load
    print("Loading embedding model...")
    embedder.load()

    vs = VectorStore()
    if vs.count() == 0:
        print("[!] No documents indexed. Run: python scripts/build_index.py")
        sys.exit(1)
    print(f"Vector store: {vs.count()} chunks")

    # Build BM25
    bm25 = BM25Index()
    coll = vs.collection
    results = coll.get(include=["documents", "metadatas"])
    if results["ids"]:
        docs = []
        for i, doc_id in enumerate(results["ids"]):
            meta = results["metadatas"][i] if results["metadatas"] else {}
            docs.append(Document(
                chunk_id=doc_id,
                text=results["documents"][i],
                source_file=meta.get("source_file", ""),
                page_number=meta.get("page_number", 1),
                chunk_index=meta.get("chunk_index", 0),
                language=meta.get("language", "en"),
            ))
        bm25.build(docs)

    # Run evaluation
    print("\nRunning evaluation on test cases...\n")
    eval_results = run_evaluation(vs, bm25)

    # Print summary
    s = eval_results["summary"]
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Test cases:              {s['total_test_cases']}")
    print(f"  Avg Faithfulness:        {s['avg_faithfulness']:.4f}  (target: >= 0.85) {'PASS' if s['faithfulness_pass'] else 'FAIL'}")
    print(f"  Avg Answer Correctness:  {s['avg_answer_correctness']:.4f}  (target: >= 0.80) {'PASS' if s['accuracy_pass'] else 'FAIL'}")
    print(f"  Avg Retrieval Latency:   {s['avg_retrieval_latency_ms']:.0f} ms")
    print(f"  Avg Generation Latency:  {s['avg_generation_latency_ms']:.0f} ms")
    print(f"  Avg Total Latency:       {s['avg_total_latency_ms']:.0f} ms")
    print(f"  Out-of-scope refusal:    {s['out_of_scope_correctly_refused']}/{s['out_of_scope_questions']}")
    print()

    # Print details
    print("=" * 60)
    print("  PER-QUESTION DETAILS")
    print("=" * 60)
    for r in eval_results["details"]:
        status = "PASS" if r["faithfulness"] >= 0.7 or r["is_out_of_scope"] else "FAIL"
        print(f"  [{status}] Q{r['index']}: {r['question'][:70]}...")
        print(f"          Faithfulness={r['faithfulness']:.3f}, "
              f"Correctness={r['answer_correctness']:.3f}, "
              f"Latency={r['retrieval_latency_ms']:.0f}+{r['generation_latency_ms']:.0f}ms")
    print()

    # Save
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=2)
    print(f"Full results saved to: {args.output}")


if __name__ == "__main__":
    main()
