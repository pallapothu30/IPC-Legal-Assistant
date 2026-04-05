from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    chunk_size: int = 700
    chunk_overlap: int = 120
    top_k: int = 5
    embedding_model_name: str = "BAAI/bge-base-en-v1.5"
    groq_model_name: str = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")
    temperature: float = 0.1
    max_tokens: int = 500
    persist_directory: Path = BASE_DIR / os.getenv("CHROMA_PERSIST_DIR", "chroma_db")
    collection_name: str = os.getenv("COLLECTION_NAME", "ipc_sections")
    default_data_path: Path = BASE_DIR / os.getenv("IPC_DATA_PATH", "data/indian_penal_code.pdf")
    groq_api_key: str | None = os.getenv("GROQ_API_KEY")
    api_host: str = os.getenv("API_HOST", "127.0.0.1")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    api_timeout: int = int(os.getenv("API_TIMEOUT", "30"))


settings = Settings()

