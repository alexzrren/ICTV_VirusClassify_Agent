"""
HMMER wrapper for extracting target genomic regions (e.g. RdRp) from query sequences.

Wraps hmmsearch + coordinate-based sequence extraction.
Reuses logic from the existing ictv_classifier pipeline.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_HMM_DIR = os.environ.get(
    "HMM_DIR",
    "/home/renzirui/Projects/Phylogenetic_Background/VHDB_RdRp_firstpass/Crop_Profile"
)


def hmmsearch_protein(protein_seq: str, hmm_path: str,
                      min_score: float = 30.0) -> list[dict]:
    """
    Run hmmsearch of a protein sequence against an HMM profile.

    Returns list of domain hits: [{start, end, score, evalue}]
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as f:
        f.write(f">query\n{protein_seq}\n")
        query_path = f.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".domtbl", delete=False) as f:
        domtbl_path = f.name

    try:
        result = subprocess.run(
            ["hmmsearch", "--domtblout", domtbl_path, "-E", "1e-3",
             "--domE", "1e-3", hmm_path, query_path],
            capture_output=True, text=True, check=False
        )
        if result.returncode != 0:
            raise RuntimeError(f"hmmsearch failed: {result.stderr[:500]}")
        return _parse_domtbl(domtbl_path, min_score)
    finally:
        Path(query_path).unlink(missing_ok=True)
        Path(domtbl_path).unlink(missing_ok=True)


def _parse_domtbl(domtbl_path: str, min_score: float) -> list[dict]:
    hits = []
    with open(domtbl_path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            if len(parts) < 23:
                continue
            try:
                score = float(parts[13])
                evalue = float(parts[12])
                env_from = int(parts[19])
                env_to = int(parts[20])
            except (ValueError, IndexError):
                continue
            if score >= min_score:
                hits.append({
                    "start": env_from - 1,  # 0-based
                    "end": env_to,
                    "score": score,
                    "evalue": evalue,
                })
    # Sort by score descending
    hits.sort(key=lambda h: -h["score"])
    return hits


def extract_hmm_region(nucleotide_seq: str, family: str,
                        hmm_dir: Optional[str] = None) -> Optional[str]:
    """
    Extract the HMM-matched protein region from a nucleotide sequence.

    Uses EMBOSS getorf to predict ORFs, then hmmsearch against the family HMM.
    Returns the best-matching protein subsequence, or None if not found.
    """
    hdir = Path(hmm_dir or _HMM_DIR)
    hmm_path = hdir / f"{family}.hmm"
    if not hmm_path.exists():
        # Try case-insensitive
        matches = list(hdir.glob(f"{family}.hmm")) + list(hdir.glob(f"{family.lower()}.hmm"))
        if not matches:
            raise FileNotFoundError(f"HMM profile not found for {family} in {hdir}")
        hmm_path = matches[0]

    # Step 1: EMBOSS getorf to predict ORFs
    with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as f:
        f.write(f">query\n{nucleotide_seq}\n")
        nuc_path = f.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".orf.fasta", delete=False) as f:
        orf_path = f.name

    try:
        result = subprocess.run(
            ["getorf", "-sequence", nuc_path, "-outseq", orf_path,
             "-find", "1", "-minsize", "300"],
            capture_output=True, text=True, check=False
        )
        if result.returncode != 0:
            raise RuntimeError(f"getorf failed: {result.stderr[:300]}")

        # Parse ORF proteins
        orf_text = Path(orf_path).read_text()
        orfs = _parse_fasta_seqs(orf_text)
        if not orfs:
            return None

        # Step 2: Find the best ORF matching the HMM
        best_protein = None
        best_score = 0.0
        for header, prot_seq in orfs.items():
            hits = hmmsearch_protein(prot_seq, str(hmm_path))
            if hits and hits[0]["score"] > best_score:
                best_score = hits[0]["score"]
                h = hits[0]
                best_protein = prot_seq[h["start"]:h["end"]]

        return best_protein
    finally:
        Path(nuc_path).unlink(missing_ok=True)
        Path(orf_path).unlink(missing_ok=True)


def _parse_fasta_seqs(text: str) -> dict[str, str]:
    seqs: dict[str, str] = {}
    cur = ""
    for line in text.splitlines():
        if line.startswith(">"):
            cur = line[1:].split()[0]
            seqs[cur] = ""
        elif cur:
            seqs[cur] += line.strip().replace("*", "")
    return seqs
