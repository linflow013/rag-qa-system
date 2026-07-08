"""Text chunking with token-aware splitting and language detection."""

import hashlib
import logging
import re
from typing import List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class Document:
    """A chunked document segment with metadata."""

    def __init__(
        self,
        chunk_id: str,
        text: str,
        source_file: str,
        page_number: int = 1,
        chunk_index: int = 0,
        language: str = "unknown",
        is_table: bool = False,
    ):
        self.chunk_id = chunk_id
        self.text = text
        self.source_file = source_file
        self.page_number = page_number
        self.chunk_index = chunk_index
        self.language = language
        self.is_table = is_table

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source_file": self.source_file,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "language": self.language,
            "is_table": self.is_table,
        }


def _detect_language(text: str) -> str:
    """Simple CN/EN detection based on character ranges."""
    cn_chars = len(re.findall(r"[一-鿿]", text))
    en_chars = len(re.findall(r"[a-zA-Z]", text))
    if cn_chars > en_chars:
        return "cn"
    return "en"


def _generate_chunk_id(source_file: str, page: int, idx: int, text: str) -> str:
    """Generate a unique chunk identifier."""
    content_hash = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
    return f"{source_file}_p{page}_c{idx}_{content_hash}"


class Chunker:
    """Splits document text into overlapping chunks with metadata."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", ". ", " ", ""],
            length_function=len,
        )
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(
        self,
        text: str,
        source_file: str,
        page_number: int = 1,
        is_table: bool = False,
    ) -> List[Document]:
        """Split text into chunks and attach metadata."""
        if not text or not text.strip():
            return []

        chunks = self.splitter.split_text(text)
        documents = []

        for i, chunk_text in enumerate(chunks):
            chunk_text = chunk_text.strip()
            if len(chunk_text) < 20:  # skip very small chunks
                continue

            chunk_id = _generate_chunk_id(source_file, page_number, i, chunk_text)
            lang = _detect_language(chunk_text)

            documents.append(Document(
                chunk_id=chunk_id,
                text=chunk_text,
                source_file=source_file,
                page_number=page_number,
                chunk_index=i,
                language=lang,
                is_table=is_table,
            ))

        return documents

    def chunk_page(
        self, page, source_file: str
    ) -> List[Document]:
        """Chunk a single page, including table content."""
        docs = self.chunk_text(
            text=page.text,
            source_file=source_file,
            page_number=page.page_number,
        )

        # Chunk table content separately
        for table in page.tables:
            table_text = "\n".join(
                " | ".join(str(cell) for cell in row)
                for row in table
            )
            table_docs = self.chunk_text(
                text=table_text,
                source_file=source_file,
                page_number=page.page_number,
                is_table=True,
            )
            docs.extend(table_docs)

        return docs
