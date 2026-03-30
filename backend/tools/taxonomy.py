"""
Taxonomy lookup tools backed by SQLite (taxonomy.db from MSL40).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional

# Default DB path relative to project root (ictv_agent/)
_DEFAULT_DB = Path(__file__).parent.parent.parent / "data" / "taxonomy.db"


def _conn(db_path: Optional[str] = None) -> sqlite3.Connection:
    path = Path(db_path) if db_path else _DEFAULT_DB
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def lookup_species(name: str, db_path: Optional[str] = None) -> list[dict]:
    """Exact or fuzzy lookup by species name. Returns list of taxonomy dicts."""
    conn = _conn(db_path)
    cur = conn.cursor()
    # Try exact first
    rows = cur.execute(
        "SELECT * FROM species WHERE species = ? COLLATE NOCASE LIMIT 10", (name,)
    ).fetchall()
    if not rows:
        # Fuzzy: contains
        rows = cur.execute(
            "SELECT * FROM species WHERE species LIKE ? COLLATE NOCASE LIMIT 10",
            (f"%{name}%",)
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def lookup_by_family(family: str, db_path: Optional[str] = None) -> list[dict]:
    """List all species in a family."""
    conn = _conn(db_path)
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT * FROM species WHERE family = ? COLLATE NOCASE ORDER BY genus, species",
        (family,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def lookup_by_genus(genus: str, db_path: Optional[str] = None) -> list[dict]:
    conn = _conn(db_path)
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT * FROM species WHERE genus = ? COLLATE NOCASE ORDER BY species",
        (genus,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def full_taxonomy(species_name: str, db_path: Optional[str] = None) -> Optional[dict]:
    """Return the full taxonomy dict for an exact species name match."""
    results = lookup_species(species_name, db_path)
    if not results:
        return None
    return results[0]


def family_summary(family: str, db_path: Optional[str] = None) -> dict:
    """Return genus count, species count, and realm for a family."""
    conn = _conn(db_path)
    cur = conn.cursor()
    row = cur.execute("""
        SELECT
            COUNT(DISTINCT genus) AS genus_count,
            COUNT(*) AS species_count,
            MAX(realm) AS realm,
            MAX("order") AS "order"
        FROM species
        WHERE family = ? COLLATE NOCASE
    """, (family,)).fetchone()
    conn.close()
    if row:
        return dict(row)
    return {}


def search_any_level(query: str, db_path: Optional[str] = None) -> list[dict]:
    """Search across species, genus, family, order columns."""
    conn = _conn(db_path)
    cur = conn.cursor()
    like = f"%{query}%"
    rows = cur.execute("""
        SELECT * FROM species
        WHERE species LIKE ? COLLATE NOCASE
           OR genus    LIKE ? COLLATE NOCASE
           OR family   LIKE ? COLLATE NOCASE
           OR "order"  LIKE ? COLLATE NOCASE
        LIMIT 20
    """, (like, like, like, like)).fetchall()
    conn.close()
    return [dict(r) for r in rows]
