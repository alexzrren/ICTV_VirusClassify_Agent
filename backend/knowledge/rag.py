"""
RAG (Retrieval-Augmented Generation) over ICTV family text documents.

Primary: keyword/TF-IDF search (no heavy ML deps).
Optional: ChromaDB with embeddings if available.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_DEFAULT_TXT_DIR = _PROJECT_ROOT.parent / "ictv_txt"

# ── Text chunking ────────────────────────────────────────────────────────────

def _clean_text(raw: str) -> str:
    """Collapse excessive whitespace from html2txt output."""
    # Replace runs of blank lines with a single newline
    text = re.sub(r"[ \t]+", " ", raw)
    text = re.sub(r"\n\s*\n", "\n", text)
    return text.strip()


def _chunk_by_sentences(text: str, chunk_size: int = 600, overlap: int = 100) -> list[str]:
    """Split text into overlapping chunks at sentence boundaries."""
    # Split on sentence endings (. or ; followed by space/newline)
    sentences = re.split(r"(?<=[.;!?])\s+", text)
    chunks = []
    current = ""
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if len(current) + len(sent) + 1 <= chunk_size:
            current = (current + " " + sent).strip()
        else:
            if current:
                chunks.append(current)
            # carry overlap from the tail of previous chunk
            if overlap > 0 and current:
                tail = current[-overlap:]
                current = tail + " " + sent
            else:
                current = sent
    if current and len(current) > 30:
        chunks.append(current)
    return chunks


# ── Document store (lazy-loaded, cached) ────────────────────────────────────

_doc_cache: Optional[dict[str, list[dict]]] = None


def _load_chunks(txt_dir: Optional[str] = None) -> dict[str, list[dict]]:
    """Load and chunk all family txt files. Returns {family: [{text, idx}]}."""
    global _doc_cache
    if _doc_cache is not None and txt_dir is None:
        return _doc_cache

    d = Path(txt_dir or _DEFAULT_TXT_DIR)
    result = {}
    for p in sorted(d.glob("*.txt")):
        family = p.stem.lower()
        raw = p.read_text(encoding="utf-8", errors="replace")
        clean = _clean_text(raw)
        chunks = _chunk_by_sentences(clean, chunk_size=600, overlap=80)
        result[family] = [{"text": c, "idx": i} for i, c in enumerate(chunks)]

    if txt_dir is None:
        _doc_cache = result
    return result


# ── TF-IDF keyword search ───────────────────────────────────────────────────

def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in re.findall(r"[a-zA-Z]{2,}", text)]


def keyword_search(query: str, n_results: int = 5,
                   txt_dir: Optional[str] = None) -> list[dict]:
    """
    BM25-like keyword search over ICTV text chunks.
    Scores by weighted term frequency with IDF boost.
    """
    family_chunks = _load_chunks(txt_dir)

    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    # Build IDF from all chunks
    total_docs = sum(len(cs) for cs in family_chunks.values())
    doc_freq: Counter = Counter()
    for chunks in family_chunks.values():
        for chunk in chunks:
            tokens_set = set(_tokenize(chunk["text"]))
            for t in tokens_set:
                doc_freq[t] += 1

    idf = {}
    for t in query_tokens:
        df = doc_freq.get(t, 0)
        idf[t] = math.log((total_docs + 1) / (df + 1)) + 1

    # Score each chunk
    results = []
    for family, chunks in family_chunks.items():
        for chunk in chunks:
            text_lower = chunk["text"].lower()
            score = 0.0
            for t in query_tokens:
                tf = text_lower.count(t)
                if tf > 0:
                    score += (1 + math.log(tf)) * idf.get(t, 1.0)
            if score > 0:
                results.append({
                    "text": chunk["text"],
                    "family": family,
                    "chunk_idx": chunk["idx"],
                    "score": round(score, 3),
                })

    results.sort(key=lambda x: -x["score"])
    return results[:n_results]


# ── Public API ───────────────────────────────────────────────────────────────

def semantic_search(query: str, n_results: int = 5,
                    vectordb_path: Optional[str] = None) -> list[dict]:
    """
    Search ICTV family documents.
    Uses keyword/TF-IDF search (always available, no external deps).
    """
    return keyword_search(query, n_results)


def search_family(family: str, query: str, n_results: int = 3,
                  txt_dir: Optional[str] = None) -> list[dict]:
    """Search within a single family's document."""
    family_chunks = _load_chunks(txt_dir)
    chunks = family_chunks.get(family.lower(), [])
    if not chunks:
        return []

    query_tokens = _tokenize(query)
    results = []
    for chunk in chunks:
        text_lower = chunk["text"].lower()
        score = sum(text_lower.count(t) for t in query_tokens)
        if score > 0:
            results.append({
                "text": chunk["text"],
                "family": family.lower(),
                "chunk_idx": chunk["idx"],
                "score": score,
            })
    results.sort(key=lambda x: -x["score"])
    return results[:n_results]
