from __future__ import annotations


class VectorstoreNotInitializedError(RuntimeError):
    """Raised when the Chroma database is not available."""


class RagConfigurationError(RuntimeError):
    """Raised when the RAG application is missing required configuration."""

