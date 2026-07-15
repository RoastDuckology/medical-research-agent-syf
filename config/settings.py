"""Central environment configuration for the project."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Application settings loaded from environment variables."""

    deepseek_api_key: str
    deepseek_base_url: str
    llm_model: str
    vector_db_path: Path
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    retrieval_top_k: int
    chroma_collection_name: str
    logs_dir: Path
    github_token: str
    arxiv_max_results: int
    github_max_results: int
    request_timeout: int


def get_settings() -> Settings:
    """Return the current application settings."""

    return Settings(
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", "").strip(),
        deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip(),
        llm_model=os.getenv("LLM_MODEL", "deepseek-chat").strip(),
        vector_db_path=Path(os.getenv("VECTOR_DB_PATH", "data/vector_db")),
        embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5").strip(),
        chunk_size=int(os.getenv("CHUNK_SIZE", "500")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "80")),
        retrieval_top_k=int(os.getenv("RETRIEVAL_TOP_K", "3")),
        chroma_collection_name=os.getenv("CHROMA_COLLECTION_NAME", "research_logs").strip(),
        logs_dir=Path(os.getenv("LOGS_DIR", "data/demo_logs")),
        github_token=os.getenv("GITHUB_TOKEN", "").strip(),
        arxiv_max_results=int(os.getenv("ARXIV_MAX_RESULTS", "5")),
        github_max_results=int(os.getenv("GITHUB_MAX_RESULTS", "5")),
        request_timeout=int(os.getenv("REQUEST_TIMEOUT", "10")),
    )
