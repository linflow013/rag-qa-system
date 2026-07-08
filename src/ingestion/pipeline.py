"""Ingestion pipeline orchestrator — raw documents to chunked Documents."""

import logging
import os
from typing import List

from src.ingestion.chunker import Chunker, Document
from src.ingestion.pdf_parser import parse_pdf_with_ocr_fallback
from src.ingestion.docx_parser import parse_docx
from config import config

logger = logging.getLogger(__name__)


def _is_scanned_page(page) -> bool:
    """Heuristic: a page with very few extractable chars is likely scanned."""
    return hasattr(page, "char_count") and page.char_count < config.ingestion.ocr_fallback_threshold


def ingest_file(filepath: str, chunker: Chunker) -> List[Document]:
    """
    Ingest a single file (PDF or DOCX) and return chunked Documents.
    """
    ext = os.path.splitext(filepath)[1].lower()
    source_file = os.path.basename(filepath)
    all_docs = []

    if ext == ".pdf":
        pages = parse_pdf_with_ocr_fallback(
            filepath, ocr_threshold=config.ingestion.ocr_fallback_threshold
        )
        for page in pages:
            docs = chunker.chunk_page(page, source_file)
            if _is_scanned_page(page):
                for d in docs:
                    d.language = "cn"  # scanned docs default to CN
            all_docs.extend(docs)

    elif ext in (".docx", ".doc"):
        pages = parse_docx(filepath)
        for page in pages:
            docs = chunker.chunk_page(page, source_file)
            all_docs.extend(docs)

    elif ext == ".txt":
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
        all_docs.extend(chunker.chunk_text(text, source_file))

    else:
        logger.warning("Unsupported file type: %s", ext)

    return all_docs


def ingest_directory(directory: str) -> List[Document]:
    """
    Walk a directory and ingest all supported files.
    Returns a flat list of chunked Documents.
    """
    chunker = Chunker(
        chunk_size=config.ingestion.chunk_size,
        chunk_overlap=config.ingestion.chunk_overlap,
    )

    all_docs = []
    file_count = 0

    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.startswith("~") or filename.startswith("."):
                continue
            filepath = os.path.join(root, filename)
            try:
                docs = ingest_file(filepath, chunker)
                all_docs.extend(docs)
                file_count += 1
                logger.info("Ingested %s → %d chunks", filename, len(docs))
            except Exception as e:
                logger.error("Failed to ingest %s: %s", filename, e)

    logger.info(
        "Ingestion complete: %d files → %d chunks (chunk_size=%d, overlap=%d)",
        file_count, len(all_docs),
        config.ingestion.chunk_size, config.ingestion.chunk_overlap,
    )
    return all_docs
