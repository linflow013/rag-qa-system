#!/usr/bin/env python
"""
Interactive terminal demo of the RAG QA system.

Usage:
    python scripts/run_demo.py
"""
import logging
import sys
import time

sys.path.insert(0, ".")

from config import config
from src.indexing.embedder import embedder
from src.indexing.vector_store import VectorStore
from src.indexing.bm25_index import BM25Index
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.sparse_retriever import SparseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.llm_client import get_llm_client
from src.generation.prompt_builder import build_prompt, build_system_prompt
from src.generation.guard import check_retrieval_quality, detect_injection, detect_language_from_results
from src.generation.citation import clean_answer, extract_citations
from src.conversation.memory import ConversationMemory
from src.logging.structured_logger import rag_logger
from src.logging.cost_tracker import CostTracker

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger("demo")


def main():
    print("=" * 60)
    print("  RAG QA System - Interactive Demo")
    print("  AcmeTech Enterprise Knowledge Base")
    print("=" * 60)
    print()

    # Initialize
    print("Loading embedding model (this may take a moment on first run)...")
    embedder.load()

    vs = VectorStore()
    if vs.count() == 0:
        print("\n[!] No documents in the index.")
        print("    Run: python scripts/build_index.py")
        sys.exit(1)

    print(f"Vector store: {vs.count()} chunks loaded.\n")

    # Build BM25 from vector store
    bm25 = BM25Index()
    from src.ingestion.chunker import Document
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

    dense = DenseRetriever(vs)
    sparse = SparseRetriever(bm25)
    hybrid = HybridRetriever(dense, sparse, rrf_k=config.retrieval.rrf_k)
    llm = get_llm_client()
    conv_memory = ConversationMemory()
    cost_tracker = CostTracker()
    session_id = "demo"

    print("Type your questions. Type 'quit' to exit, 'clear' to reset conversation.\n")
    print("Example questions:")
    print("  - How many days of annual leave do employees get?")
    print("  - What is the API authentication method?")
    print("  - 员工每年有多少天年假？")
    print()

    turn = 0
    while True:
        try:
            question = input(f"[Turn {turn + 1}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if question.lower() == "clear":
            conv_memory.clear(session_id)
            turn = 0
            print("[Conversation cleared]\n")
            continue

        turn += 1
        total_start = time.time()

        # Guard: injection check
        if detect_injection(question):
            print("\n[!] Prompt injection detected. Please rephrase.\n")
            continue

        # Retrieval
        r_start = time.time()
        retrieval_results = hybrid.retrieve(question, top_k=5)
        r_latency = (time.time() - r_start) * 1000

        # Guard: retrieval quality
        is_ok, refusal = check_retrieval_quality(
            retrieval_results, score_threshold=config.retrieval.score_threshold
        )
        if not is_ok:
            print(f"\n{refusal}\n")
            continue

        # Generation
        history = conv_memory.get_history(session_id)
        lang = detect_language_from_results(retrieval_results)
        system_prompt = build_system_prompt(lang)
        prompt = build_prompt(question, retrieval_results, history)

        gen_result = llm.generate(prompt=prompt, system_prompt=system_prompt, history=history)
        total_latency = (time.time() - total_start) * 1000

        answer = clean_answer(gen_result.text)
        sources = list(set(r.metadata.get("source_file", "") for r in retrieval_results))
        citations = extract_citations(answer, sources)

        # Print answer
        print(f"\n{answer}\n")
        if citations:
            print("Sources:")
            for c in citations:
                print(f"  - {c.source_file}, page {c.page_number}")
        print(f"\n[Latency: {total_latency:.0f}ms | Tokens: {gen_result.input_tokens} in / {gen_result.output_tokens} out | Model: {gen_result.model_name}]")
        print()

        # Update memory and cost
        conv_memory.add_turn(session_id, question, answer)
        cost_tracker.record(gen_result.input_tokens, gen_result.output_tokens)

        # Log
        rag_logger.log_full(
            session_id=session_id,
            question=question,
            answer=answer,
            retrieval_latency_ms=r_latency,
            generation_latency_ms=gen_result.latency_ms,
            total_latency_ms=total_latency,
            num_chunks=len(retrieval_results),
            top_score=retrieval_results[0].score if retrieval_results else 0,
            input_tokens=gen_result.input_tokens,
            output_tokens=gen_result.output_tokens,
            model_name=gen_result.model_name,
            turn_number=turn,
            citations=[c.to_dict() for c in citations],
        )


if __name__ == "__main__":
    main()
