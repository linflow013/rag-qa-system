"""Cross-encoder reranker for improving retrieval precision."""

import logging
from typing import List

from src.indexing.vector_store import SearchResult
from config import config

logger = logging.getLogger(__name__)


class Reranker:
    """Re-ranks retrieval results using a cross-encoder model."""

    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.retrieval.reranker_model
        self._model = None
        self.loaded = False

    def load(self):
        if self.loaded:
            return
        try:
            from sentence_transformers import CrossEncoder
            logger.info("Loading reranker model: %s ...", self.model_name)
            self._model = CrossEncoder(self.model_name)
            self.loaded = True
            logger.info("Reranker model loaded.")
        except Exception as e:
            logger.warning("Failed to load reranker: %s. Reranking disabled.", e)
            self._model = None

    def rerank(
        self, query: str, candidates: List[SearchResult]
    ) -> List[SearchResult]:
        """
        Re-rank candidates using cross-encoder relevance scoring.
        Returns re-sorted list.
        """
        if not self.loaded:
            self.load()

        if self._model is None or len(candidates) <= 1:
            return candidates

        pairs = [(query, c.text) for c in candidates]
        scores = self._model.predict(pairs)

        # Assign new scores and re-sort
        for result, score in zip(candidates, scores):
            result.score = float(score)

        candidates.sort(key=lambda x: x.score, reverse=True)
        logger.debug("Reranked %d candidates", len(candidates))
        return candidates
