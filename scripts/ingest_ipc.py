#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.config import settings
from backend.services.ingestion_service import ingest_source_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest IPC data into ChromaDB.")
    parser.add_argument(
        "--input",
        type=str,
        default=str(settings.default_data_path),
        help="Path to the IPC PDF or text file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    sections_processed, chunks_stored = ingest_source_file(args.input)
    print(f"Ingestion complete for {args.input}")
    print(f"Sections processed: {sections_processed}")
    print(f"Chunks stored: {chunks_stored}")
    print(f"Collection name: {settings.collection_name}")
    print(f"Persist directory: {settings.persist_directory}")
