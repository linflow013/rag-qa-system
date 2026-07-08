"""Central configuration for the RAG QA system."""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


@dataclass
class IngestionConfig:
    chunk_size: int = 512
    chunk_overlap: int = 64
    ocr_fallback_threshold: int = 50  # chars per page, below which OCR is used
    ocr_lang: str = "ch"  # PaddleOCR language


@dataclass
class EmbeddingConfig:
    model_name: str = "BAAI/bge-m3"
    device: str = "cpu"
    dense_dim: int = 1024
    query_instruction: str = (
        "Represent this sentence for searching relevant passages: "
    )


@dataclass
class RetrievalConfig:
    top_k: int = 5
    rrf_k: int = 60  # RRF fusion parameter
    score_threshold: float = 0.01  # below this, trigger refusal (RRF scores are small, ~0.016-0.033)
    reranker_enabled: bool = False
    reranker_model: str = "BAAI/bge-reranker-v2-m3"


@dataclass
class GenerationConfig:
    model: str = "deepseek-chat"
    temperature: float = 0.0
    max_tokens: int = 1024
    max_history_turns: int = 5


@dataclass
class LLMConfig:
    provider: str = "deepseek"
    api_key: str = field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", ""))
    base_url: str = field(
        default_factory=lambda: os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    )
    model: str = "deepseek-chat"


@dataclass
class Config:
    ingestion: IngestionConfig = field(default_factory=IngestionConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)

    # Paths
    data_dir: str = os.path.join(os.path.dirname(__file__), "data")
    raw_dir: str = os.path.join(data_dir, "raw")
    processed_dir: str = os.path.join(data_dir, "processed")
    sample_dir: str = os.path.join(data_dir, "sample")
    chroma_dir: str = os.path.join(os.path.dirname(__file__), "chroma_db")
    logs_dir: str = os.path.join(os.path.dirname(__file__), "logs")

    def __post_init__(self):
        for d in [self.data_dir, self.raw_dir, self.processed_dir,
                  self.sample_dir, self.chroma_dir, self.logs_dir]:
            os.makedirs(d, exist_ok=True)


# Singleton config
config = Config()
