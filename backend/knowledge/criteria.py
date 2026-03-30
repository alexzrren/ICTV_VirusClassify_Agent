"""
Load and query the extracted ICTV demarcation criteria knowledge base.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

_DEFAULT_PATH = Path(__file__).parent.parent.parent / "data" / "criteria.json"

_cache: Optional[dict] = None


def _load(path: Optional[str] = None) -> dict:
    global _cache
    if _cache is not None and path is None:
        return _cache
    p = Path(path) if path else _DEFAULT_PATH
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    if path is None:
        _cache = data
    return data


def get_criteria(family: str, path: Optional[str] = None) -> Optional[dict]:
    """
    Return the demarcation criteria dict for a family.
    family: case-insensitive, e.g. "Coronaviridae" or "coronaviridae"
    """
    db = _load(path)
    key = family.lower()
    return db.get(key) or db.get(family)


def list_families(path: Optional[str] = None) -> list[str]:
    """Return all family names in the criteria DB."""
    return list(_load(path).keys())


def get_demarcation_summary(family: str, level: str = "species",
                             path: Optional[str] = None) -> str:
    """
    Return a human-readable summary of demarcation criteria for a family at a given level.

    level: "species", "genus", or "subfamily"
    """
    crit = get_criteria(family, path)
    if not crit:
        return f"No criteria found for {family}."

    key = f"{level}_demarcation"
    d = crit.get(key)
    if not d:
        return f"No {level}-level demarcation criteria available for {family}."

    method = d.get("primary_method") or "unspecified"
    regions = ", ".join(d.get("regions") or []) or "unspecified"
    thresholds = d.get("thresholds") or {}
    desc = d.get("description") or ""

    parts = [
        f"Family: {family}",
        f"Level: {level}",
        f"Method: {method}",
        f"Genomic region(s): {regions}",
    ]
    if thresholds:
        thr_str = "; ".join(f"{k}={v}" for k, v in thresholds.items())
        parts.append(f"Thresholds: {thr_str}")
    if desc:
        parts.append(f"Description: {desc}")

    return "\n".join(parts)
