from __future__ import annotations

import shutil
from pathlib import Path
from typing import Sequence

from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import settings
from embeddings.hf_embeddings import get_embedding_model


def get_vectorstore() -> Chroma:
    """Load the persistent Chroma vector store."""
    return Chroma(
        collection_name=settings.collection_name,
        persist_directory=str(settings.persist_directory),
        embedding_function=get_embedding_model(),
    )


def build_vectorstore(documents: Sequence[Document], reset: bool = True) -> Chroma:
    """Create and persist the Chroma vector store from IPC chunks."""
    persist_dir = Path(settings.persist_directory)
    if reset and persist_dir.exists():
        shutil.rmtree(persist_dir)

    persist_dir.mkdir(parents=True, exist_ok=True)

    vectorstore = Chroma.from_documents(
        documents=list(documents),
        embedding=get_embedding_model(),
        collection_name=settings.collection_name,
        persist_directory=str(persist_dir),
    )
    return vectorstore
