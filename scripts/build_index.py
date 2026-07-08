#!/usr/bin/env python
"""
Build the RAG index from documents in data/raw/.

Usage:
    python scripts/build_index.py
    python scripts/build_index.py --data-dir ./data/raw --rebuild
"""
import argparse
import logging
import sys
import time

sys.path.insert(0, ".")

from config import config
from src.ingestion.pipeline import ingest_directory
from src.indexing.embedder import embedder
from src.indexing.vector_store import VectorStore
from src.indexing.bm25_index import BM25Index

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger("build_index")


def main():
    parser = argparse.ArgumentParser(description="Build RAG index from documents")
    parser.add_argument("--data-dir", default=config.raw_dir, help="Directory with source documents")
    parser.add_argument("--rebuild", action="store_true", help="Delete existing index and rebuild")
    parser.add_argument("--chunk-size", type=int, default=512, help="Chunk size in tokens")
    parser.add_argument("--chunk-overlap", type=int, default=64, help="Chunk overlap in tokens")
    args = parser.parse_args()

    # Override config
    config.ingestion.chunk_size = args.chunk_size
    config.ingestion.chunk_overlap = args.chunk_overlap

    overall_start = time.time()

    # Step 1: Ingest documents
    logger.info("=== Step 1: Ingesting documents from %s ===", args.data_dir)
    documents = ingest_directory(args.data_dir)

    if not documents:
        logger.error("No documents found in %s. Please add PDF/DOCX files.", args.data_dir)
        sys.exit(1)

    logger.info("Ingested %d documents (%d chunks)", len(set(d.source_file for d in documents)), len(documents))

    # Step 2: Generate embeddings
    logger.info("=== Step 2: Generating embeddings ===")
    embedder.load()
    texts = [doc.text for doc in documents]
    embeddings = embedder.encode_dense(texts)

    # Step 3: Build vector store
    logger.info("=== Step 3: Building vector store ===")
    vs = VectorStore()
    if args.rebuild:
        try:
            vs.delete_collection()
        except Exception:
            pass
    vs.add_documents(documents, embeddings)

    # Step 4: Build BM25 index
    logger.info("=== Step 4: Building BM25 index ===")
    bm25 = BM25Index()
    bm25.build(documents)

    total_time = time.time() - overall_start

    logger.info("=== Done ===")
    logger.info("Total documents: %d", len(set(d.source_file for d in documents)))
    logger.info("Total chunks: %d", len(documents))
    logger.info("Vector store size: %d", vs.count())
    logger.info("Total time: %.1f seconds", total_time)


if __name__ == "__main__":
    main()
