"""
BLAST/DIAMOND search tools for initial family identification.

Supports:
- blastn (nucleotide query vs nucleotide DB)
- blastp (protein query vs protein DB)
- diamond blastp (faster protein search)

DB paths are read from environment variables or use defaults under data/db/.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


_PROJECT_ROOT = Path(__file__).parent.parent.parent
_DATA_DB = _PROJECT_ROOT / "data" / "db"

BLASTN_DB = os.environ.get("BLASTN_DB", str(_DATA_DB / "blastn_ref"))
DIAMOND_DB = os.environ.get("DIAMOND_DB", str(_DATA_DB / "diamond_ref.dmnd"))


@dataclass
class BlastHit:
    query_id: str
    subject_id: str
    pident: float       # % identity
    length: int
    evalue: float
    bitscore: float
    qcovs: float = 0.0  # query coverage %
    stitle: str = ""    # subject title / description
    # Parsed extras
    family: str = ""
    accession: str = ""


def _parse_tabular(output: str, fields: list[str]) -> list[BlastHit]:
    hits = []
    for line in output.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split("\t")
        d = {f: parts[i] for i, f in enumerate(fields) if i < len(parts)}
        try:
            hit = BlastHit(
                query_id=d.get("qseqid", ""),
                subject_id=d.get("sseqid", ""),
                pident=float(d.get("pident", 0)),
                length=int(d.get("length", 0)),
                evalue=float(d.get("evalue", 1)),
                bitscore=float(d.get("bitscore", 0)),
                qcovs=float(d.get("qcovs", 0)),
                stitle=d.get("stitle", ""),
            )
            hits.append(hit)
        except (ValueError, KeyError):
            continue
    return hits


BLAST_OUTFMT = "6 qseqid sseqid pident length evalue bitscore qcovs stitle"
BLAST_FIELDS = ["qseqid", "sseqid", "pident", "length", "evalue", "bitscore", "qcovs", "stitle"]


def blastn(sequence: str, db: Optional[str] = None, max_hits: int = 10,
           evalue: float = 1e-5) -> list[BlastHit]:
    """Run blastn for a nucleotide query sequence."""
    db_path = db or BLASTN_DB
    if not Path(str(db_path) + ".nhr").exists() and not Path(str(db_path) + ".nin").exists():
        raise FileNotFoundError(
            f"BLASTN database not found at {db_path}. "
            "Run scripts/build_blast_db.py first."
        )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as f:
        f.write(f">query\n{sequence}\n")
        query_path = f.name
    try:
        # dc-megablast is required for cross-species virus searches (~60-80% identity).
        # megablast (default) uses word_size=28 and misses remote homologs.
        result = subprocess.run(
            ["blastn", "-query", query_path, "-db", db_path,
             "-task", "dc-megablast",
             "-outfmt", BLAST_OUTFMT, "-max_target_seqs", str(max_hits),
             "-evalue", str(evalue), "-num_threads", "4"],
            capture_output=True, text=True, check=False
        )
        if result.returncode not in (0, 1):  # 1 = no hits
            raise RuntimeError(f"blastn error: {result.stderr[:500]}")
        return _parse_tabular(result.stdout, BLAST_FIELDS)
    finally:
        Path(query_path).unlink(missing_ok=True)


def diamond_blastp(sequence: str, db: Optional[str] = None, max_hits: int = 10,
                   evalue: float = 1e-5) -> list[BlastHit]:
    """Run DIAMOND blastp for a protein query sequence."""
    db_path = db or DIAMOND_DB
    if not Path(str(db_path)).exists():
        raise FileNotFoundError(
            f"DIAMOND database not found at {db_path}. "
            "Run scripts/build_blast_db.py first."
        )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as f:
        f.write(f">query\n{sequence}\n")
        query_path = f.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False) as f:
        out_path = f.name
    try:
        result = subprocess.run(
            ["diamond", "blastp", "-q", query_path, "-d", db_path,
             "-o", out_path, "--outfmt", "6",
             "qseqid", "sseqid", "pident", "length", "evalue", "bitscore", "qcovs", "stitle",
             "-k", str(max_hits), "--evalue", str(evalue),
             "--quiet", "--threads", "4"],
            capture_output=True, text=True, check=False
        )
        if result.returncode != 0:
            raise RuntimeError(f"diamond error: {result.stderr[:500]}")
        output = Path(out_path).read_text()
        return _parse_tabular(output, BLAST_FIELDS)
    finally:
        Path(query_path).unlink(missing_ok=True)
        Path(out_path).unlink(missing_ok=True)


def identify_family_from_hits(hits: list[BlastHit],
                               accession_to_family: dict[str, str]) -> Optional[str]:
    """
    Given a list of BLAST hits and an accession→family mapping,
    return the most likely family (majority vote among top hits).
    """
    if not hits:
        return None
    family_votes: dict[str, float] = {}
    for hit in hits[:5]:
        acc = hit.subject_id.split(".")[0]  # strip version
        fam = accession_to_family.get(acc) or accession_to_family.get(hit.subject_id)
        if fam:
            family_votes[fam] = family_votes.get(fam, 0) + hit.bitscore
    if not family_votes:
        return None
    return max(family_votes, key=lambda k: family_votes[k])
