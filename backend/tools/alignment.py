"""
Sequence alignment and pairwise identity tools using MAFFT.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Union


def run_mafft_pairwise(seq1: str, seq2: str, is_protein: bool = False) -> tuple[str, str]:
    """
    Align two sequences with MAFFT (--auto) and return (aligned_seq1, aligned_seq2).

    seq1, seq2: raw sequences (no FASTA headers), uppercase or lowercase.
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as f:
        f.write(f">seq1\n{seq1}\n>seq2\n{seq2}\n")
        fasta_path = Path(f.name)

    out_path = fasta_path.with_suffix(".afa")
    cmd = ["mafft", "--auto", "--quiet", str(fasta_path)]
    if is_protein:
        cmd.insert(1, "--amino")

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    fasta_path.unlink(missing_ok=True)

    if result.returncode != 0:
        raise RuntimeError(f"MAFFT failed: {result.stderr[:500]}")

    # Parse aligned sequences from stdout
    seqs: dict[str, str] = {}
    cur_name = ""
    for line in result.stdout.splitlines():
        if line.startswith(">"):
            cur_name = line[1:].strip()
            seqs[cur_name] = ""
        else:
            seqs[cur_name] += line.strip()

    if "seq1" not in seqs or "seq2" not in seqs:
        raise RuntimeError("MAFFT output missing seq1/seq2")

    return seqs["seq1"], seqs["seq2"]


def pairwise_identity(seq1: str, seq2: str, is_protein: bool = False) -> float:
    """
    Compute pairwise identity (fraction of identical positions / alignment length,
    excluding columns where BOTH positions are gaps).

    Returns a float in [0, 1].
    """
    aln1, aln2 = run_mafft_pairwise(seq1, seq2, is_protein=is_protein)
    aln1 = aln1.upper()
    aln2 = aln2.upper()

    if len(aln1) != len(aln2):
        raise ValueError("Aligned sequences have different lengths")

    identical = 0
    total = 0
    for a, b in zip(aln1, aln2):
        if a == "-" and b == "-":
            continue
        total += 1
        if a == b:
            identical += 1

    return identical / total if total > 0 else 0.0


def pairwise_identity_no_align(seq1: str, seq2: str) -> float:
    """
    Quick identity for pre-aligned sequences of the same length.
    No MAFFT call needed.
    """
    if len(seq1) != len(seq2):
        raise ValueError("Sequences must be the same length for pre-aligned identity")
    s1 = seq1.upper()
    s2 = seq2.upper()
    identical = sum(1 for a, b in zip(s1, s2) if a == b and a != "-")
    total = sum(1 for a, b in zip(s1, s2) if not (a == "-" and b == "-"))
    return identical / total if total > 0 else 0.0


def parse_fasta(text: str) -> dict[str, str]:
    """Parse FASTA text → {header: sequence}."""
    seqs: dict[str, str] = {}
    cur = ""
    for line in text.splitlines():
        line = line.strip()
        if line.startswith(">"):
            cur = line[1:]
            seqs[cur] = ""
        elif cur:
            seqs[cur] += line
    return seqs
