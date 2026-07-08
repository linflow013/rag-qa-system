# Design Note: RAG QA System

## Key Design Decisions and Trade-offs

**Why a custom pipeline instead of LangChain or LlamaIndex?**

LangChain and LlamaIndex provide high-level abstractions that speed up prototyping but obscure the retrieval-augmentation loop. For a system where faithfulness and citations are the primary quality targets, every intermediate step — prompt assembly, retrieval fusion, citation validation — must be auditable and debuggable. A custom pipeline with ~1,200 lines of application code gives full control over these steps without hidden prompts or default behaviors. The trade-off is more initial implementation effort, but the result is a codebase where each component (ingestion, retrieval, generation) can be understood, tested, and evolved independently.

**Why hybrid retrieval (dense + sparse with RRF fusion)?**

Pure dense retrieval (vector similarity) excels at semantic matching but can miss exact keyword matches for terms like "PIPL" or "PostgreSQL 15+". BM25 captures these precise lexical matches but lacks semantic understanding for paraphrased queries. Reciprocal Rank Fusion (RRF, k=60) combines both signals without requiring weight tuning — a parameter that often needs per-corpus adjustment. The cost is roughly 2x the retrieval latency (two searches instead of one), but this stays well within the 10-second budget.

**Why BAAI/bge-m3 for embeddings?**

The model is natively multilingual (100+ languages including strong CN/EN performance), outputs both dense and sparse vectors from a single model, and runs on CPU at acceptable speed for a demo corpus. The alternative — separate models for English and Chinese embeddings — would complicate the architecture and require language detection before retrieval.

**CPU-only embedding trade-off.**

Embedding on CPU takes 30-60 seconds for ~250 chunks (a 100-page document). For a demo corpus under 500 documents, this is acceptable. A production system would benefit from GPU acceleration, but the architecture doesn't change — the embedding model is a swappable component.

**Latency budget breakdown (target: <10s for 90% of requests):**

| Component | Estimated (CPU) |
|-----------|----------------|
| Query embedding | 100-200ms |
| Dense retrieval (ChromaDB) | 50-100ms |
| BM25 retrieval | 20-50ms |
| RRF fusion | <10ms |
| Reranker (optional) | 300-500ms |
| LLM generation (API) | 2-5s |
| **Total** | **3-6s** |

The 10-second target is achievable even with the reranker enabled, assuming the LLM API responds within 5 seconds.

**Cost estimation methodology.**

Using DeepSeek API pricing ($0.14/1M input, $0.28/1M output tokens), with typical usage of ~1,800 input tokens (5 chunks × 200 tokens + 300 prompt overhead + 500 history) and ~200 output tokens per query. Estimated cost: ~$0.31 per 1,000 calls at baseline settings, scaling proportionally with top_k.
