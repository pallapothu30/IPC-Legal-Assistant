from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader


def load_ipc_source(file_path: str | Path) -> str:
    """Load raw text from a PDF or UTF-8 text file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"IPC source file not found: {path}")

    if path.suffix.lower() == ".pdf":
        pages = PyPDFLoader(str(path)).load()
        return "\n".join(page.page_content for page in pages)

    if path.suffix.lower() in {".txt", ".md"}:
        docs = TextLoader(str(path), encoding="utf-8").load()
        return "\n".join(doc.page_content for doc in docs)

    raise ValueError("Unsupported file type. Please provide a .pdf, .txt, or .md file.")

