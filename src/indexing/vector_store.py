"""ChromaDB vector store for dense embedding storage and search."""

import logging
from typing import Dict, List, Optional

import numpy as np

from config import config

logger = logging.getLogger(__name__)


class SearchResult:
    """Result from a vector similarity search."""

    def __init__(self, chunk_id: str, text: str, score: float, metadata: dict):
        self.chunk_id = chunk_id
        self.text = text
        self.score = score
        self.metadata = metadata

    def __repr__(self):
        return (
            f"SearchResult(chunk_id={self.chunk_id!r}, "
            f"score={self.score:.4f}, source={self.metadata.get('source_file', '?')})"
        )


class VectorStore:
    """Manages ChromaDB collection: insert chunks and search by embedding."""

    def __init__(self, collection_name: str = "knowledge_base"):
        self.collection_name = collection_name
        self._client = None
        self._collection = None

    @property
    def client(self):
        if self._client is None:
            import chromadb
            self._client = chromadb.PersistentClient(
                path=config.chroma_dir,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )
        return self._client

    @property
    def collection(self):
        if self._collection is None:
            self._collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def add_documents(
        self,
        documents: List,
        embeddings: np.ndarray,
    ):
        """Add chunked documents with their embeddings to the store."""
        if len(documents) == 0:
            return

        ids = [doc.chunk_id for doc in documents]
        texts = [doc.text for doc in documents]
        metadatas = [doc.to_dict() for doc in documents]

        # ChromaDB needs list-of-lists for embeddings
        emb_list = embeddings.tolist()

        batch_size = 100
        for i in range(0, len(ids), batch_size):
            batch_slice = slice(i, i + batch_size)
            self.collection.add(
                ids=ids[batch_slice],
                documents=texts[batch_slice],
                metadatas=metadatas[batch_slice],
                embeddings=emb_list[batch_slice],
            )

        logger.info("Added %d documents to ChromaDB collection '%s'",
                     len(documents), self.collection_name)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter_metadata: Optional[dict] = None,
    ) -> List[SearchResult]:
        """Search for closest documents by cosine similarity."""
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=filter_metadata,
            include=["documents", "metadatas", "distances"],
        )

        search_results = []
        if results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i]
                # ChromaDB returns cosine distance; convert to similarity
                score = 1.0 - distance

                search_results.append(SearchResult(
                    chunk_id=chunk_id,
                    text=results["documents"][0][i],
                    score=score,
                    metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                ))

        return search_results

    def count(self) -> int:
        """Return the number of documents in the collection."""
        return self.collection.count()

    def delete_collection(self):
        """Delete the entire collection (rebuild)."""
        self.client.delete_collection(self.collection_name)
        self._collection = None
        logger.info("Deleted collection '%s'", self.collection_name)

    def get_stats(self) -> dict:
        """Return collection statistics."""
        return {
            "collection_name": self.collection_name,
            "count": self.count(),
        }
