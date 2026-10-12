"""Central environment configuration for the project."""

from __future__ import annotations

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
    llm_fallback_models: tuple[str, ...]
    llm_max_tokens: int
    llm_retry_attempts: int
    llm_failure_cooldown_seconds: int
    vector_db_path: Path
    embedding_model: str
    chunk_size: int
    chunk_overlap: int
    retrieval_top_k: int
    retrieval_min_score: float
    retrieval_domain_guard: bool
    chroma_collection_name: str
    logs_dir: Path
    extra_markdown_files: tuple[Path, ...]
    paper_notes_dir: Path
    papers_dir: Path
    paper_text_dir: Path
    enable_pdf_ingestion: bool
    memory_db_path: Path
    github_token: str
    arxiv_max_results: int
    github_max_results: int
    request_timeout: int
    context_budget_tokens: int


def get_settings() -> Settings:
    """Return the current application settings."""

    extra_markdown_files = _parse_path_list(
        os.getenv("EXTRA_MARKDOWN_FILES", "实验进度记录.md")
    )

    return Settings(
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", "").strip(),
        deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").strip(),
        llm_model=os.getenv("LLM_MODEL", "deepseek-chat").strip(),
        llm_fallback_models=_parse_string_list(
            os.getenv("LLM_FALLBACK_MODELS", "deepseek-chat")
        ),
        llm_max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1200")),
        llm_retry_attempts=int(os.getenv("LLM_RETRY_ATTEMPTS", "1")),
        llm_failure_cooldown_seconds=int(os.getenv("LLM_FAILURE_COOLDOWN_SECONDS", "300")),
        vector_db_path=Path(os.getenv("VECTOR_DB_PATH", "data/vector_db")),
        embedding_model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5").strip(),
        chunk_size=int(os.getenv("CHUNK_SIZE", "500")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "80")),
        retrieval_top_k=int(os.getenv("RETRIEVAL_TOP_K", "3")),
        retrieval_min_score=float(os.getenv("RETRIEVAL_MIN_SCORE", "0")),
        retrieval_domain_guard=_parse_bool(os.getenv("RETRIEVAL_DOMAIN_GUARD", "1")),
        chroma_collection_name=os.getenv("CHROMA_COLLECTION_NAME", "research_logs").strip(),
        logs_dir=Path(os.getenv("LOGS_DIR", "data/logs")),
        extra_markdown_files=extra_markdown_files,
        paper_notes_dir=Path(os.getenv("PAPER_NOTES_DIR", "data/paper_notes")),
        papers_dir=Path(os.getenv("PAPERS_DIR", "data/papers")),
        paper_text_dir=Path(os.getenv("PAPER_TEXT_DIR", "data/paper_text")),
        enable_pdf_ingestion=_parse_bool(os.getenv("ENABLE_PDF_INGESTION", "1")),
        memory_db_path=Path(os.getenv("MEMORY_DB_PATH", "data/memory/agent_memory.sqlite3")),
        github_token=os.getenv("GITHUB_TOKEN", "").strip(),
        arxiv_max_results=int(os.getenv("ARXIV_MAX_RESULTS", "5")),
        github_max_results=int(os.getenv("GITHUB_MAX_RESULTS", "5")),
        request_timeout=int(os.getenv("REQUEST_TIMEOUT", "20")),
        context_budget_tokens=int(os.getenv("CONTEXT_BUDGET_TOKENS", "2800")),
    )


def _parse_path_list(value: str) -> tuple[Path, ...]:
    """Parse semicolon-separated paths from an environment variable."""

    paths: list[Path] = []
    for raw_path in value.split(";"):
        normalized = raw_path.strip().strip('"')
        if normalized:
            paths.append(Path(normalized))
    return tuple(paths)


def _parse_string_list(value: str) -> tuple[str, ...]:
    items: list[str] = []
    for raw_item in value.split(";"):
        normalized = raw_item.strip().strip('"')
        if normalized:
            items.append(normalized)
    return tuple(items)


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}
