"""Dense retriever — vector similarity search."""

import logging
from typing import List

import numpy as np

from src.indexing.embedder import embedder
from src.indexing.vector_store import SearchResult, VectorStore

logger = logging.getLogger(__name__)


class DenseRetriever:
    """Retrieves documents by dense vector similarity."""

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 5) -> List[SearchResult]:
        """Encode query and search the vector store."""
        query_embedding = embedder.encode_single(query, is_query=True)
        results = self.vector_store.search(query_embedding, top_k=top_k)
        logger.debug("Dense retrieval: %d results for query (top_k=%d)",
                     len(results), top_k)
        return results
