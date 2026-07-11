#!/usr/bin/env python
"""Ingest data/sources/ into the persistent Chroma collection at ./chroma_db.

Usage:
    python scripts/ingest.py [--source-dir data/sources] [--persist-dir ./chroma_db]
        [--extracted-dir data/extracted] [--skip-extraction]

Requires OPENAI_API_KEY (from the environment or a .env file). Idempotent:
re-running skips already-ingested chunks and already-extracted documents.

Two independent outputs per run:
1. Structured extraction — one gpt-4o-mini call per new source document pulls
   an ExtractedFluidProfile into data/extracted/{slug}.json when the document
   describes a specific fluid product (regulatory background docs are skipped).
2. Chunking + embedding into Chroma for RAG, as before.
"""

import argparse
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

from coolant_copilot.extraction import run_extraction
from coolant_copilot.ingestion import (
    DEFAULT_COLLECTION,
    DEFAULT_PERSIST_DIR,
    ingest,
    load_documents,
)

DEFAULT_EXTRACTED_DIR = "data/extracted"


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", default="data/sources")
    parser.add_argument("--persist-dir", default=DEFAULT_PERSIST_DIR)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--extracted-dir", default=DEFAULT_EXTRACTED_DIR)
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Only chunk/embed; do not run structured profile extraction.",
    )
    args = parser.parse_args()

    docs = load_documents(Path(args.source_dir))

    if not args.skip_extraction:
        summary = run_extraction(docs, Path(args.extracted_dir))
        print(
            f"Extraction: {len(summary['extracted'])} extracted, "
            f"{len(summary['skipped'])} skipped, "
            f"{len(summary['cached'])} already processed."
        )
        for source, n_props in summary["extracted"].items():
            print(f"  extracted {source}: {n_props} propert{'y' if n_props == 1 else 'ies'}")
        for source in summary["skipped"]:
            print(f"  skipped {source}: not a fluid product document")

    counts = ingest(
        source_dir=args.source_dir,
        embeddings=OpenAIEmbeddings(model="text-embedding-3-small"),
        persist_dir=args.persist_dir,
        collection_name=args.collection,
    )
    print(
        f"Embedding: {counts['total']} chunks: {counts['added']} added, "
        f"{counts['skipped']} already present."
    )


if __name__ == "__main__":
    main()
