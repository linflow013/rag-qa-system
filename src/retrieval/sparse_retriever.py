"""Sparse retriever — BM25 keyword-based search."""

import logging
from typing import List

from src.indexing.bm25_index import BM25Index
from src.indexing.vector_store import SearchResult

logger = logging.getLogger(__name__)


class SparseRetriever:
    """Retrieves documents by BM25 keyword matching."""

    def __init__(self, bm25_index: BM25Index):
        self.bm25_index = bm25_index

    def retrieve(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Search the BM25 index and convert to SearchResults."""
        hits = self.bm25_index.search(query, top_k=top_k)
        results = []

        for idx, score in hits:
            doc = self.bm25_index._documents[idx]
            # Normalize BM25 score to [0, 1] range (approximate)
            normalized_score = min(score / 20.0, 1.0)
            results.append(SearchResult(
                chunk_id=doc.chunk_id,
                text=doc.text,
                score=normalized_score,
                metadata=doc.to_dict(),
            ))

        logger.debug("Sparse retrieval: %d results for query (top_k=%d)",
                     len(results), top_k)
        return results
