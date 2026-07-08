"""BM25 sparse index for keyword-based retrieval."""

import logging
import re
from typing import List

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class BM25Index:
    """Sparse lexical index using BM25 for keyword matching."""

    def __init__(self):
        self._index = None
        self._documents = []
        self._tokenized_corpus = []

    @staticmethod
    def _tokenize_cn(text: str) -> List[str]:
        """Tokenize Chinese text using jieba."""
        try:
            import jieba
            return list(jieba.cut(text))
        except ImportError:
            # Fallback: character-level tokenization
            return list(text)

    @staticmethod
    def _tokenize_en(text: str) -> List[str]:
        """Tokenize English text — lowercase, split on non-alpha."""
        return re.findall(r"[a-zA-Z0-9]+", text.lower())

    @classmethod
    def tokenize(cls, text: str, language: str = "cn") -> List[str]:
        """Tokenize text based on detected language."""
        tokens = []
        if language == "cn":
            tokens = cls._tokenize_cn(text)
        else:
            tokens = cls._tokenize_en(text)

        # Also add English tokens if mixed
        if language == "cn":
            tokens.extend(cls._tokenize_en(text))

        # Deduplicate while preserving order
        seen = set()
        result = []
        for t in tokens:
            if t.strip() and t not in seen:
                result.append(t.strip())
                seen.add(t)
        return result

    def build(self, documents: List):
        """Build the BM25 index from chunked documents."""
        self._documents = documents
        self._tokenized_corpus = []

        for doc in documents:
            tokens = self.tokenize(doc.text, language=doc.language)
            self._tokenized_corpus.append(tokens)

        self._index = BM25Okapi(self._tokenized_corpus)
        logger.info("BM25 index built with %d documents", len(documents))

    def search(self, query: str, top_k: int = 5) -> List[tuple]:
        """
        Search the index.
        Returns list of (document_index, bm25_score).
        """
        if self._index is None:
            return []

        # Tokenize query — try both CN and EN
        query_tokens = self._tokenize_cn(query) + self._tokenize_en(query)

        if not query_tokens:
            return []

        scores = self._index.get_scores(query_tokens)

        # Get top-k indices
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(idx, float(score)) for idx, score in ranked[:top_k] if score > 0]
