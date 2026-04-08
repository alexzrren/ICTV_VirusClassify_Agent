"""
Local result cache backed by SQLite.

Sequences are identified by SHA-256 of the cleaned (header-stripped,
uppercase, whitespace-removed) nucleotide/protein string.  When the same
sequence is submitted again the cached ClassifyResult is returned
immediately, skipping the full agent pipeline.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .models import ClassifyResult

DB_PATH = Path(__file__).parent.parent / "data" / "cache.db"


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cache (
            seq_hash    TEXT PRIMARY KEY,
            query_id    TEXT,
            fasta       TEXT,
            result_json TEXT NOT NULL,
            created_at  TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def _clean_seq(fasta: str) -> str:
    """Strip FASTA headers and whitespace, uppercase — gives a canonical sequence."""
    lines = fasta.strip().splitlines()
    seq_lines = [l.strip() for l in lines if not l.startswith(">")]
    return "".join(seq_lines).upper().replace(" ", "")


def seq_hash(fasta: str) -> str:
    """SHA-256 hex digest of the cleaned sequence."""
    return hashlib.sha256(_clean_seq(fasta).encode()).hexdigest()


def cache_get(fasta: str) -> Optional[ClassifyResult]:
    """Look up a cached result for the given FASTA input."""
    h = seq_hash(fasta)
    try:
        conn = _connect()
        row = conn.execute(
            "SELECT result_json FROM cache WHERE seq_hash = ?", (h,)
        ).fetchone()
        conn.close()
        if row:
            return ClassifyResult.model_validate_json(row[0])
    except Exception:
        pass
    return None


def cache_put(fasta: str, result: ClassifyResult) -> None:
    """Store a classification result in the cache."""
    h = seq_hash(fasta)
    try:
        conn = _connect()
        conn.execute(
            "INSERT OR REPLACE INTO cache (seq_hash, query_id, fasta, result_json, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                h,
                result.query_id,
                fasta[:500],  # store truncated fasta for display
                result.model_dump_json(),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def cache_history(limit: int = 20) -> list[dict]:
    """Return recent cached results (newest first)."""
    try:
        conn = _connect()
        rows = conn.execute(
            "SELECT seq_hash, query_id, fasta, result_json, created_at "
            "FROM cache ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        conn.close()
        results = []
        for h, qid, fasta_snippet, rj, ts in rows:
            r = json.loads(rj)
            tax = r.get("taxonomy", {})
            results.append({
                "seq_hash": h,
                "query_id": qid or r.get("query_id", "?"),
                "fasta_preview": fasta_snippet[:80],
                "family": tax.get("family"),
                "genus": tax.get("genus"),
                "species": tax.get("species"),
                "confidence": r.get("confidence"),
                "created_at": ts,
            })
        return results
    except Exception:
        return []


def cache_get_by_hash(h: str) -> Optional[ClassifyResult]:
    """Retrieve a cached result by its sequence hash."""
    try:
        conn = _connect()
        row = conn.execute(
            "SELECT result_json FROM cache WHERE seq_hash = ?", (h,)
        ).fetchone()
        conn.close()
        if row:
            return ClassifyResult.model_validate_json(row[0])
    except Exception:
        pass
    return None
