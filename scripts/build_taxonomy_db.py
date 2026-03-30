#!/usr/bin/env python3
"""
Build SQLite taxonomy database from ICTV Master Species List (MSL40) Excel.

Usage:
    python scripts/build_taxonomy_db.py \
        --msl ../../ICTV_Master_Species_List_2024_MSL40.v2.xlsx \
        --output data/taxonomy.db
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import openpyxl

TAXONOMY_LEVELS = [
    "Realm", "Subrealm", "Kingdom", "Subkingdom",
    "Phylum", "Subphylum", "Class", "Subclass",
    "Order", "Suborder", "Family", "Subfamily",
    "Genus", "Subgenus", "Species",
]

CREATE_SPECIES_TABLE = """
CREATE TABLE IF NOT EXISTS species (
    id          INTEGER PRIMARY KEY,
    sort        INTEGER,
    realm       TEXT,
    subrealm    TEXT,
    kingdom     TEXT,
    subkingdom  TEXT,
    phylum      TEXT,
    subphylum   TEXT,
    class       TEXT,
    subclass    TEXT,
    "order"     TEXT,
    suborder    TEXT,
    family      TEXT,
    subfamily   TEXT,
    genus       TEXT,
    subgenus    TEXT,
    species     TEXT NOT NULL,
    ictv_id     TEXT,
    genome      TEXT,
    last_change TEXT,
    msl_change  TEXT,
    proposal    TEXT
);
"""

CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_species_name ON species(species COLLATE NOCASE);",
    "CREATE INDEX IF NOT EXISTS idx_family ON species(family COLLATE NOCASE);",
    "CREATE INDEX IF NOT EXISTS idx_genus ON species(genus COLLATE NOCASE);",
    "CREATE INDEX IF NOT EXISTS idx_ictv_id ON species(ictv_id);",
]


def load_msl(msl_path: Path) -> list[dict]:
    wb = openpyxl.load_workbook(str(msl_path), read_only=True, data_only=True)
    ws = wb["MSL"]
    rows = ws.iter_rows(values_only=True)
    header = [str(c).strip() if c else "" for c in next(rows)]

    col = {name: i for i, name in enumerate(header)}
    records = []
    for row in rows:
        if not row[col.get("Species", 15)]:
            continue
        records.append({
            "sort":        row[col.get("Sort", 0)],
            "realm":       row[col.get("Realm", 1)],
            "subrealm":    row[col.get("Subrealm", 2)],
            "kingdom":     row[col.get("Kingdom", 3)],
            "subkingdom":  row[col.get("Subkingdom", 4)],
            "phylum":      row[col.get("Phylum", 5)],
            "subphylum":   row[col.get("Subphylum", 6)],
            "class":       row[col.get("Class", 7)],
            "subclass":    row[col.get("Subclass", 8)],
            "order":       row[col.get("Order", 9)],
            "suborder":    row[col.get("Suborder", 10)],
            "family":      row[col.get("Family", 11)],
            "subfamily":   row[col.get("Subfamily", 12)],
            "genus":       row[col.get("Genus", 13)],
            "subgenus":    row[col.get("Subgenus", 14)],
            "species":     row[col.get("Species", 15)],
            "ictv_id":     row[col.get("ICTV_ID", 16)],
            "genome":      row[col.get("Genome", 17)],
            "last_change": str(row[col.get("Last Change", 18)]) if row[col.get("Last Change", 18)] else None,
            "msl_change":  str(row[col.get("MSL of Last Change", 19)]) if row[col.get("MSL of Last Change", 19)] else None,
            "proposal":    str(row[col.get("Proposal for Last Change ", 20)]) if col.get("Proposal for Last Change ") is not None and row[col.get("Proposal for Last Change ", 20)] else None,
        })
    wb.close()
    return records


def build_db(msl_path: Path, db_path: Path) -> None:
    print(f"Loading MSL from {msl_path} ...")
    records = load_msl(msl_path)
    print(f"  {len(records)} species loaded.")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute(CREATE_SPECIES_TABLE)
    for idx_sql in CREATE_INDEXES:
        cur.execute(idx_sql)

    cur.executemany("""
        INSERT INTO species (
            sort, realm, subrealm, kingdom, subkingdom,
            phylum, subphylum, class, subclass, "order", suborder,
            family, subfamily, genus, subgenus, species,
            ictv_id, genome, last_change, msl_change, proposal
        ) VALUES (
            :sort, :realm, :subrealm, :kingdom, :subkingdom,
            :phylum, :subphylum, :class, :subclass, :order, :suborder,
            :family, :subfamily, :genus, :subgenus, :species,
            :ictv_id, :genome, :last_change, :msl_change, :proposal
        )
    """, records)

    conn.commit()
    count = cur.execute("SELECT COUNT(*) FROM species").fetchone()[0]
    conn.close()
    print(f"  Written {count} rows to {db_path}")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Build SQLite taxonomy DB from MSL40 Excel.")
    ap.add_argument("--msl", required=True, help="Path to ICTV_Master_Species_List_*.xlsx")
    ap.add_argument("--output", default="data/taxonomy.db", help="Output SQLite path")
    args = ap.parse_args(argv)

    build_db(Path(args.msl), Path(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
