from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Iterable, Sequence

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.core.config import settings

SECTION_PATTERN = re.compile(
    r"(?im)^\s*(Section\s+)?(?P<number>\d+[A-Z]?)\s*[\.\-:)]?\s*(?P<title>[^\n]{3,160})\n(?P<body>.*?)(?=^\s*(?:Section\s+)?\d+[A-Z]?\s*[\.\-:)]?\s*[^\n]{3,160}\n|\Z)",
    re.DOTALL,
)


def load_source(file_path: str | Path) -> str:
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


def normalize_text(raw_text: str) -> str:
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_sections(raw_text: str, source_name: str) -> list[Document]:
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

    return [
        Document(
            page_content=f"Section {match.group('number').strip()}: {match.group('title').strip(' .:-')}\n{match.group('body').strip()}".strip(),
            metadata={
                "source": source_name,
                "section_number": match.group("number").strip(),
                "section_title": match.group("title").strip(" .:-"),
            },
        )
        for match in matches
    ]


def chunk_sections(section_documents: Iterable[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunked_docs: list[Document] = []
    for doc in section_documents:
        for index, chunk in enumerate(splitter.split_documents([doc])):
            chunk.metadata["chunk_id"] = f"{doc.metadata['section_number']}_{index}"
            chunked_docs.append(chunk)
    return chunked_docs


def get_embedding_model() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def get_vectorstore() -> Chroma:
    return Chroma(
        collection_name=settings.collection_name,
        persist_directory=str(settings.persist_directory),
        embedding_function=get_embedding_model(),
    )


def build_vectorstore(documents: Sequence[Document], reset: bool = True) -> Chroma:
    if reset and settings.persist_directory.exists():
        shutil.rmtree(settings.persist_directory)

    settings.persist_directory.mkdir(parents=True, exist_ok=True)
    return Chroma.from_documents(
        documents=list(documents),
        embedding=get_embedding_model(),
        collection_name=settings.collection_name,
        persist_directory=str(settings.persist_directory),
    )


def ingest_source_file(file_path: str | Path) -> tuple[int, int]:
    input_path = Path(file_path)
    raw_text = load_source(input_path)
    section_docs = extract_sections(raw_text, source_name=input_path.name)
    chunked_docs = chunk_sections(section_docs)
    build_vectorstore(chunked_docs)
    return len(section_docs), len(chunked_docs)

