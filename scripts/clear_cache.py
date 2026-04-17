#!/usr/bin/env python3
"""
Clear the ICTV Agent classification cache.

The agent caches every successful classification keyed by SHA-256 of the input
sequence. Use this script to wipe the cache when you want to re-run all
classifications from scratch (e.g. after upgrading the agent or VMR DB).

Usage:
    python scripts/clear_cache.py            # clear results cache
    python scripts/clear_cache.py --all      # also drop history table
    python scripts/clear_cache.py --dry-run  # show what would be deleted
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
CACHE_DB = PROJECT / "data" / "cache.db"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--all", action="store_true",
                   help="Also drop history table (default: only clear cache)")
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would be deleted without doing it")
    p.add_argument("--db", default=str(CACHE_DB),
                   help=f"Cache DB path (default: {CACHE_DB})")
    args = p.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"Cache DB does not exist: {db_path}", file=sys.stderr)
        print("Nothing to clear.", file=sys.stderr)
        return 0

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    tables = [r[0] for r in cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]

    if not tables:
        print(f"DB has no tables: {db_path}", file=sys.stderr)
        return 0

    print(f"Cache DB: {db_path}")
    counts = {}
    for t in tables:
        try:
            n = cur.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        except sqlite3.OperationalError:
            n = 0
        counts[t] = n
        print(f"  {t:20s}: {n} rows")

    if args.dry_run:
        print("\n[dry-run] No changes made.")
        return 0

    targets = ["cache"]
    if args.all:
        targets = tables  # everything

    cleared = 0
    for t in targets:
        if t not in tables:
            continue
        n_before = counts[t]
        cur.execute(f"DELETE FROM {t}")
        cleared += n_before
        print(f"  Cleared {n_before} rows from {t}")

    conn.commit()
    cur.execute("VACUUM")
    conn.close()

    print(f"\nDone. Removed {cleared} rows total.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
