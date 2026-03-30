#!/usr/bin/env python3
"""
Build ChromaDB vector index from ICTV family text documents.

Usage:
    python scripts/build_vectordb.py \
        --txt-dir ../ictv_txt \
        --output data/vectordb \
        [--chunk-size 800] [--overlap 100]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """Split text into overlapping chunks of ~chunk_size characters."""
    # Split by double newline (paragraphs), then merge until chunk_size
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if len(p.strip()) > 30]
    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) < chunk_size:
            current += " " + para if current else para
        else:
            if current:
                chunks.append(current)
            # Start new chunk with overlap from end of previous
            current = current[-overlap:] + " " + para if current and overlap > 0 else para
    if current:
        chunks.append(current)
    return chunks


def build_vectordb(txt_dir: Path, output: Path, chunk_size: int, overlap: int) -> None:
    try:
        import chromadb
    except ImportError:
        print("ERROR: chromadb not installed. Run: pip install chromadb", file=sys.stderr)
        sys.exit(1)

    output.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(output))

    # Delete existing collection if rebuilding
    try:
        client.delete_collection("ictv_docs")
    except Exception:
        pass

    coll = client.create_collection(
        "ictv_docs",
        metadata={"hnsw:space": "cosine"},
    )

    ids, docs, metas = [], [], []
    txt_files = sorted(txt_dir.glob("*.txt"))
    for txt_path in txt_files:
        family = txt_path.stem.lower()
        text = txt_path.read_text(encoding="utf-8", errors="replace")
        chunks = chunk_text(text, chunk_size, overlap)
        for i, chunk in enumerate(chunks):
            ids.append(f"{family}_chunk_{i:04d}")
            docs.append(chunk)
            metas.append({"family": family, "chunk_id": i})
        print(f"  {family}: {len(chunks)} chunks")

    # Add in batches of 100
    batch = 100
    for i in range(0, len(ids), batch):
        coll.add(
            ids=ids[i:i+batch],
            documents=docs[i:i+batch],
            metadatas=metas[i:i+batch],
        )

    total = coll.count()
    print(f"\nDone: {total} chunks indexed in {output}")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Build ChromaDB vector index from ICTV txt files.")
    ap.add_argument("--txt-dir", default="../ictv_txt")
    ap.add_argument("--output", default="data/vectordb")
    ap.add_argument("--chunk-size", type=int, default=800)
    ap.add_argument("--overlap", type=int, default=100)
    args = ap.parse_args(argv)

    build_vectordb(Path(args.txt_dir), Path(args.output), args.chunk_size, args.overlap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
