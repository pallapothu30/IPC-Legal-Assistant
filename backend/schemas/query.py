from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The IPC-related question")
    chat_history: list[dict] = Field(default_factory=list, description="Previous chat messages")


class SourceDocument(BaseModel):
    content: str = Field(..., description="The text content of the document")
    section_number: str | None = Field(None, description="IPC section number")
    section_title: str | None = Field(None, description="IPC section title")
    source: str = Field(..., description="Source reference")


class QueryResponse(BaseModel):
    answer: str = Field(..., description="Grounded answer from IPC context")
    source_documents: list[SourceDocument] = Field(..., description="Retrieved supporting chunks")
    relevant_sections: list[str] = Field(..., description="Relevant IPC section numbers")


class ConfigResponse(BaseModel):
    model: str = Field(..., description="LLM model name")
    embedding: str = Field(..., description="Embedding model name")
    top_k: int = Field(..., description="Retriever top-k")
    chroma_db: str = Field(..., description="Chroma database location")

