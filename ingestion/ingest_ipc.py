from __future__ import annotations

import argparse
from pathlib import Path

from config import settings
from ingestion.loader import load_ipc_source
from ingestion.preprocess import chunk_sections, extract_sections
from vectorstore.chroma_store import build_vectorstore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest IPC data into ChromaDB.")
    parser.add_argument(
        "--input",
        type=str,
        default=str(settings.default_data_path),
        help="Path to the IPC PDF or text file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)

    raw_text = load_ipc_source(input_path)
    section_docs = extract_sections(raw_text, source_name=input_path.name)
    chunked_docs = chunk_sections(section_docs)
    vectorstore = build_vectorstore(chunked_docs)

    print(f"Ingestion complete for {input_path.name}")
    print(f"Sections processed: {len(section_docs)}")
    print(f"Chunks stored: {len(chunked_docs)}")
    print(f"Collection name: {settings.collection_name}")
    print(f"Persist directory: {settings.persist_directory}")


if __name__ == "__main__":
    main()
