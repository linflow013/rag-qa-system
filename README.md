# RAG QA System

Retrieval-Augmented Generation (RAG) question-answering system for enterprise knowledge bases. Supports bilingual (CN/EN) document retrieval, multi-turn dialogue, citation-backed answers, and structured logging.

## Quick Start

### 1. Prerequisites

- Python 3.10+
- pip

### 2. Install Dependencies

```bash
cd rag-qa-system
pip install -r requirements.txt
```

### 3. Configure LLM Access

Copy the environment template and add your API key:

```bash
cp .env.example .env
```

Edit `.env` with your DeepSeek API key (or OpenAI key):

```
DEEPSEEK_API_KEY=sk-your-key-here
```

Get a key at: https://platform.deepseek.com
OR
You can use my deepseek api by the way, it will be mentioned in the email i send.


### 4. Generate Sample Data

```bash
python scripts/generate_sample_data.py
```

This creates 4 bilingual DOCX documents in `data/raw/`:
- Employee Handbook (员工手册)
- Compliance & Regulatory Guide (合规与监管指南)
- Technical Specifications (技术规范)
- Platform Architecture Overview (平台架构概述)

### 5. Build the Index

```bash
python scripts/build_index.py
```

This ingests documents, generates embeddings, and builds both the vector store (ChromaDB) and BM25 index.

### 6. Run the Demo

```bash
# Interactive terminal demo
python scripts/run_demo.py

# OR: Start the API server
python -m src.api.main
# Then open http://localhost:8000/docs for the Swagger UI
```

## Project Structure

```
rag-qa-system/
|-- README.md
|-- DESIGN_NOTE.md              # Design rationale (200-500 words)
|-- requirements.txt
|-- .env.example
|-- config.py                   # Central configuration
|
|-- data/
|   |-- raw/                    # Source documents
|   |-- sample/                 # Test documents
|
|-- src/
|   |-- ingestion/              # PDF/DOCX parsing, OCR, chunking
|   |-- indexing/               # Embedding model, vector store, BM25
|   |-- retrieval/              # Dense, sparse, hybrid (RRF), reranker
|   |-- generation/             # LLM client, prompts, citations, guards
|   |-- conversation/           # Session memory for multi-turn
|   |-- evaluation/             # RAGAS metrics, test cases
|   |-- logging/                # Structured JSON logging, cost tracking
|   |-- api/                    # FastAPI server
|
|-- scripts/
|   |-- generate_sample_data.py
|   |-- build_index.py
|   |-- run_demo.py
|   |-- run_evaluation.py
|   |-- cost_sensitivity.py
|
|-- tests/                      # Unit tests
|-- logs/                       # Structured log output
|-- chroma_db/                  # ChromaDB persistent storage
```

## API Endpoints

### `POST /ask`
Ask a question and get a citation-backed answer.

```json
{
  "question": "How many days of annual leave?",
  "session_id": "optional-session-id",
  "top_k": 5
}
```

### `GET /health`
Health check with index statistics.

### `POST /clear-session`
Clear conversation history for a session.

### `GET /stats`
System statistics including index size, cost, and active sessions.

## Scripts

| Script | Purpose |
|--------|---------|
| `generate_sample_data.py` | Create sample bilingual DOCX documents |
| `build_index.py` | Build vector store and BM25 index |
| `run_demo.py` | Interactive terminal QA demo |
| `run_evaluation.py` | Run RAG evaluation suite |
| `cost_sensitivity.py` | Cost/quality trade-off analysis |

## Configuration

All tunable parameters are in `config.py`:

- `chunk_size` / `chunk_overlap`: Text splitting parameters (default: 512/64)
- `top_k`: Number of documents to retrieve (default: 5)
- `temperature`: LLM generation temperature (default: 0.0)
- `reranker_enabled`: Toggle cross-encoder reranker (default: False)
- `score_threshold`: Minimum retrieval score for answering (default: 0.01)

## Sample Logs

Structured logs are written to `logs/rag_YYYY-MM-DD.jsonl`. Each line is a JSON object:

```json
{
  "timestamp": "2026-07-06T10:30:00",
  "type": "full",
  "session_id": "abc123",
  "question": "How many days of annual leave?",
  "answer": "Full-time employees are entitled to 15 working days...[redacted]",
  "retrieval_latency_ms": 150.5,
  "generation_latency_ms": 3200.0,
  "total_latency_ms": 3350.5,
  "num_chunks_retrieved": 5,
  "top_score": 0.87,
  "input_tokens": 1650,
  "output_tokens": 180,
  "model_name": "deepseek-chat",
  "turn_number": 1,
  "citations": [{"source_file": "employee_handbook.docx", "page_number": 1}]
}
```

PII (phone numbers, ID numbers, emails) is automatically redacted in logs.
