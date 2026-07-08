"""FastAPI route handlers for the RAG QA system."""

import logging
import time
from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse

from src.api.schemas import (
    AskRequest,
    AskResponse,
    CitationResponse,
    ErrorResponse,
    HealthResponse,
)
from src.conversation.memory import ConversationMemory
from src.conversation.session import memory
from src.generation.guard import (
    check_retrieval_quality,
    detect_injection,
    detect_language_from_results,
)
from src.generation.llm_client import get_llm_client
from src.generation.prompt_builder import build_prompt, build_system_prompt
from src.generation.citation import clean_answer, extract_citations
from src.indexing.embedder import embedder
from src.indexing.vector_store import VectorStore
from src.indexing.bm25_index import BM25Index
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.sparse_retriever import SparseRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.logging.structured_logger import rag_logger
from src.logging.cost_tracker import CostTracker
from config import config

logger = logging.getLogger(__name__)

router = APIRouter()

# ─── Service singletons (initialized lazily) ────────────────────────

_vector_store: Optional[VectorStore] = None
_bm25_index: Optional[BM25Index] = None
_hybrid_retriever: Optional[HybridRetriever] = None
_llm_client = None
_cost_tracker: Optional[CostTracker] = None


def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


def get_hybrid_retriever() -> HybridRetriever:
    global _hybrid_retriever, _bm25_index
    if _hybrid_retriever is None:
        vs = get_vector_store()
        dense = DenseRetriever(vs)

        # BM25 index needs to be built; check if there's data
        if _bm25_index is None:
            _bm25_index = BM25Index()
            # Try loading existing documents from vector store
            try:
                coll = vs.collection
                results = coll.get(include=["documents", "metadatas"])
                if results["ids"]:
                    # Reconstruct minimal documents for BM25
                    from src.ingestion.chunker import Document
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
                    if docs:
                        _bm25_index.build(docs)
            except Exception as e:
                logger.warning("Could not load BM25 index: %s", e)

        sparse = SparseRetriever(_bm25_index)
        _hybrid_retriever = HybridRetriever(dense, sparse, rrf_k=config.retrieval.rrf_k)
    return _hybrid_retriever


def get_llm():
    global _llm_client
    if _llm_client is None:
        _llm_client = get_llm_client()
    return _llm_client


def get_cost_tracker() -> CostTracker:
    global _cost_tracker
    if _cost_tracker is None:
        _cost_tracker = CostTracker()
    return _cost_tracker


# ─── Endpoints ──────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        vs = get_vector_store()
        return HealthResponse(
            status="healthy",
            index_size=vs.count(),
        )
    except Exception as e:
        return HealthResponse(status=f"degraded: {e}", index_size=0)


@router.post("/ask", response_model=AskResponse)
async def ask_question(req: AskRequest):
    """
    Main RAG QA endpoint.
    Accepts a question, retrieves relevant documents, generates an answer with citations.
    Supports multi-turn conversation via session_id.
    """
    total_start = time.time()
    session_id = memory.get_or_create(req.session_id or "default")
    session_data = memory._store.get(session_id, {})
    turn_number = session_data.get("turn_count", 0) + 1

    # ── Guard 1: Prompt injection check ──
    if detect_injection(req.question):
        rag_logger.log_error(session_id, req.question, "injection_detected",
                             "Prompt injection pattern matched")
        raise HTTPException(
            status_code=400,
            detail="Your question contains patterns that are not allowed. "
                   "Please rephrase and try again."
        )

    # ── Step 1: Retrieval ──
    retrieval_start = time.time()
    try:
        retriever = get_hybrid_retriever()
        retrieval_results = retriever.retrieve(req.question, top_k=req.top_k or 5)
    except Exception as e:
        logger.error("Retrieval failed: %s", e)
        retrieval_results = []

    retrieval_latency = (time.time() - retrieval_start) * 1000

    # ── Guard 2: Retrieval quality check ──
    is_ok, refusal_msg = check_retrieval_quality(
        retrieval_results, score_threshold=config.retrieval.score_threshold
    )
    if not is_ok:
        rag_logger.log_error(session_id, req.question, "low_relevance", refusal_msg)
        return AskResponse(
            answer=refusal_msg,
            citations=[],
            session_id=session_id,
            latency_ms=(time.time() - total_start) * 1000,
            token_usage={"input_tokens": 0, "output_tokens": 0, "model": "none"},
            turn_number=turn_number,
        )

    top_score = retrieval_results[0].score if retrieval_results else 0
    rag_logger.log_retrieval(
        session_id, req.question,
        len(retrieval_results), top_score, retrieval_latency,
    )

    # ── Step 2: Generation ──
    generation_start = time.time()
    try:
        llm = get_llm()
        history = memory.get_history(session_id, max_turns=config.generation.max_history_turns)
        lang = detect_language_from_results(retrieval_results)
        system_prompt = build_system_prompt(lang)
        prompt = build_prompt(req.question, retrieval_results, history)

        gen_result = llm.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            history=history,
        )
    except Exception as e:
        logger.error("Generation failed: %s", e)
        gen_latency = (time.time() - generation_start) * 1000
        rag_logger.log_error(session_id, req.question, "generation_error", str(e))
        raise HTTPException(status_code=500, detail=f"Generation failed: {e}")

    gen_latency = gen_result.latency_ms
    total_latency = (time.time() - total_start) * 1000

    # ── Step 3: Citations ──
    retrieved_sources = list(set(
        r.metadata.get("source_file", "") for r in retrieval_results
    ))
    citations = extract_citations(gen_result.text, retrieved_sources)
    answer = clean_answer(gen_result.text)

    # ── Step 4: Update memory ──
    memory.add_turn(session_id, req.question, answer)

    # ── Step 5: Track cost ──
    tracker = get_cost_tracker()
    tracker.record(gen_result.input_tokens, gen_result.output_tokens)

    # ── Step 6: Structured logging ──
    rag_logger.log_full(
        session_id=session_id,
        question=req.question,
        answer=answer,
        retrieval_latency_ms=retrieval_latency,
        generation_latency_ms=gen_latency,
        total_latency_ms=total_latency,
        num_chunks=len(retrieval_results),
        top_score=top_score,
        input_tokens=gen_result.input_tokens,
        output_tokens=gen_result.output_tokens,
        model_name=gen_result.model_name,
        turn_number=turn_number,
        citations=[c.to_dict() for c in citations],
    )

    return AskResponse(
        answer=answer,
        citations=[CitationResponse(**c.to_dict()) for c in citations],
        session_id=session_id,
        latency_ms=total_latency,
        token_usage={
            "input_tokens": gen_result.input_tokens,
            "output_tokens": gen_result.output_tokens,
            "model": gen_result.model_name,
        },
        turn_number=turn_number,
    )


@router.post("/clear-session")
async def clear_session(session_id: str):
    """Clear a conversation session."""
    memory.clear(session_id)
    return {"status": "cleared", "session_id": session_id}


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a document for indexing (placeholder)."""
    return JSONResponse(
        status_code=501,
        content={"error": "Document upload and re-indexing not yet implemented."},
    )


@router.get("/stats")
async def get_stats():
    """Return system statistics."""
    vs = get_vector_store()
    tracker = get_cost_tracker()
    return {
        "index": vs.get_stats(),
        "cost": tracker.stats(),
        "conversations": memory.stats(),
    }
