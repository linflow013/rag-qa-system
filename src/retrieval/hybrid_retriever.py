"""Hybrid retriever — Reciprocal Rank Fusion of dense + sparse results."""

import logging
from typing import Dict, List

from src.indexing.vector_store import SearchResult
from src.retrieval.dense_retriever import DenseRetriever
from src.retrieval.sparse_retriever import SparseRetriever
from config import config

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Combines dense (vector) and sparse (BM25) retrieval using
    Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        sparse_retriever: SparseRetriever,
        rrf_k: int = 60,
    ):
        self.dense = dense_retriever
        self.sparse = sparse_retriever
        self.rrf_k = rrf_k

    @staticmethod
    def _rrf_score(rank: int, k: int = 60) -> float:
        """RRF score for a document at a given rank."""
        return 1.0 / (k + rank)

    def retrieve(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """
        Retrieve using hybrid search:
        1. Get dense results (top_k * 2 for wider recall)
        2. Get sparse results (top_k * 2)
        3. Fuse via RRF
        4. Return top_k results
        """
        candidate_k = top_k * 2

        dense_results = self.dense.retrieve(query, top_k=candidate_k)
        sparse_results = self.sparse.retrieve(query, top_k=candidate_k)

        # RRF fusion
        fused: Dict[str, SearchResult] = {}
        rrf_scores: Dict[str, float] = {}

        # Add dense scores
        for rank, result in enumerate(dense_results, start=1):
            rrf_scores[result.chunk_id] = rrf_scores.get(result.chunk_id, 0) + self._rrf_score(rank, self.rrf_k)
            if result.chunk_id not in fused:
                fused[result.chunk_id] = result

        # Add sparse scores
        for rank, result in enumerate(sparse_results, start=1):
            rrf_scores[result.chunk_id] = rrf_scores.get(result.chunk_id, 0) + self._rrf_score(rank, self.rrf_k)
            if result.chunk_id not in fused:
                fused[result.chunk_id] = result

        # Sort by RRF score descending
        sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)

        final_results = []
        for chunk_id in sorted_ids[:top_k]:
            result = fused[chunk_id]
            result.score = rrf_scores[chunk_id]  # replace score with fused RRF score
            final_results.append(result)

        logger.info(
            "Hybrid retrieval: dense=%d, sparse=%d, fused=%d, final=%d",
            len(dense_results), len(sparse_results), len(fused), len(final_results),
        )

        return final_results
