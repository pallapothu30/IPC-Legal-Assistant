from __future__ import annotations

from langchain_huggingface import HuggingFaceEmbeddings

from config import settings


def get_embedding_model() -> HuggingFaceEmbeddings:
    """Return the BGE embedding model configured for similarity search."""
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

