from __future__ import annotations

import re
from typing import Iterable

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings

SECTION_PATTERN = re.compile(
    r"(?im)^\s*(Section\s+)?(?P<number>\d+[A-Z]?)\s*[\.\-:)]?\s*(?P<title>[^\n]{3,160})\n(?P<body>.*?)(?=^\s*(?:Section\s+)?\d+[A-Z]?\s*[\.\-:)]?\s*[^\n]{3,160}\n|\Z)",
    re.DOTALL,
)


def normalize_text(raw_text: str) -> str:
    """Normalize whitespace while preserving section-like line boundaries."""
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_sections(raw_text: str, source_name: str) -> list[Document]:
    """Extract IPC sections and preserve section metadata for retrieval."""
    normalized = normalize_text(raw_text)
    matches = list(SECTION_PATTERN.finditer(normalized))

    if not matches:
        return [
            Document(
                page_content=normalized,
                metadata={
                    "source": source_name,
                    "section_number": "Unknown",
                    "section_title": "Unstructured IPC Text",
                },
            )
        ]

    documents: list[Document] = []
    for match in matches:
        section_number = match.group("number").strip()
        section_title = match.group("title").strip(" .:-")
        body = match.group("body").strip()

        content = f"Section {section_number}: {section_title}\n{body}".strip()
        documents.append(
            Document(
                page_content=content,
                metadata={
                    "source": source_name,
                    "section_number": section_number,
                    "section_title": section_title,
                },
            )
        )

    return documents


def chunk_sections(section_documents: Iterable[Document]) -> list[Document]:
    """Split section documents into retrievable chunks while preserving metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunked_docs: list[Document] = []
    for doc in section_documents:
        chunks = splitter.split_documents([doc])
        for index, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = f"{doc.metadata['section_number']}_{index}"
            chunked_docs.append(chunk)
    return chunked_docs
