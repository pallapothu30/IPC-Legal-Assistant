from __future__ import annotations

from langchain_core.vectorstores import VectorStoreRetriever

from config import settings
from vectorstore.chroma_store import get_vectorstore


def get_ipc_retriever() -> VectorStoreRetriever:
    """Return the IPC retriever configured for top-k similarity search."""
    vectorstore = get_vectorstore()
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": settings.top_k},
    )

