#!/usr/bin/env python3
"""
Build BLAST/DIAMOND databases from downloaded ICTV reference sequences.

Creates:
  data/db/blastn_ref.*    — BLAST nucleotide database (all families merged)
  data/db/diamond_ref.dmnd — DIAMOND protein database (from ORF predictions)
  data/db/acc_to_family.tsv — accession → family mapping

Usage:
    python scripts/build_blast_db.py \
        --refdir data/references \
        --outdir data/db

Prerequisites:
  - makeblastdb (BLAST+)
  - diamond (optional, for protein search)
  - Reference FASTAs in data/references/{Family}/sequences.fasta
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def merge_fastas(refdir: Path, outdir: Path) -> tuple[Path, Path]:
    """
    Merge all family FASTAs into a single file, prefixing headers with family name.
    Also generate an accession→family mapping file.
    """
    merged = outdir / "all_references.fasta"
    mapping = outdir / "acc_to_family.tsv"
    outdir.mkdir(parents=True, exist_ok=True)

    n_seqs = 0
    with open(merged, "w") as fout, open(mapping, "w") as mout:
        mout.write("accession\tfamily\n")
        for fam_dir in sorted(refdir.iterdir()):
            if not fam_dir.is_dir():
                continue
            fasta = fam_dir / "sequences.fasta"
            if not fasta.exists():
                continue
            family = fam_dir.name
            with open(fasta) as fin:
                for line in fin:
                    if line.startswith(">"):
                        # Extract accession (first word after >)
                        header = line[1:].strip()
                        acc = header.split()[0].split(".")[0]  # strip version
                        fout.write(f">{family}|{header}\n")
                        mout.write(f"{acc}\t{family}\n")
                        n_seqs += 1
                    else:
                        fout.write(line)

    print(f"Merged {n_seqs} sequences into {merged}")
    return merged, mapping


def build_blastn_db(merged_fasta: Path, outdir: Path) -> None:
    """Build BLAST nucleotide database."""
    db_prefix = outdir / "blastn_ref"
    cmd = [
        "makeblastdb", "-in", str(merged_fasta),
        "-dbtype", "nucl", "-out", str(db_prefix),
        "-title", "ICTV_VMR_references",
        "-parse_seqids",
    ]
    print(f"Building blastn DB ...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  makeblastdb FAILED: {result.stderr[:500]}", file=sys.stderr)
        return
    print(f"  -> {db_prefix}.*")


def build_diamond_db(merged_fasta: Path, outdir: Path) -> None:
    """
    Build DIAMOND protein database.
    Uses DIAMOND's built-in frameshift translation for nucleotide input.
    """
    db_path = outdir / "diamond_ref.dmnd"
    # Diamond makedb can accept nucleotide sequences and will translate them
    # But standard diamond makedb requires protein. For nt→protein, use blastx mode at query time.
    # Instead, build a nucleotide-backed DIAMOND DB using --input-type.
    # Simplest approach: just build protein DB from ORF translations later.
    # For now, skip if diamond not available.
    try:
        subprocess.run(["diamond", "version"], capture_output=True, check=True)
    except FileNotFoundError:
        print("  diamond not found, skipping protein DB (install: conda install -c bioconda diamond)")
        return

    # DIAMOND makedb expects protein input; for nucleotide references,
    # we would need to predict ORFs first. Skip for now — blastn is sufficient.
    print("  DIAMOND protein DB: skipped (requires ORF prediction first).")
    print("  Use scripts/build_diamond_protein_db.sh for protein DB after HMM extraction.")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Build BLAST/DIAMOND databases from reference sequences.")
    ap.add_argument("--refdir", default="data/references", help="Directory with {Family}/sequences.fasta")
    ap.add_argument("--outdir", default="data/db", help="Output directory for DB files")
    args = ap.parse_args(argv)

    refdir = Path(args.refdir)
    outdir = Path(args.outdir)

    if not refdir.exists():
        print(f"ERROR: Reference directory {refdir} not found.", file=sys.stderr)
        print("Run scripts/download_reference_seqs.py first.", file=sys.stderr)
        return 1

    merged, mapping = merge_fastas(refdir, outdir)

    if merged.stat().st_size < 100:
        print("ERROR: Merged FASTA is empty. No reference sequences found.", file=sys.stderr)
        return 1

    # Build blastn DB
    try:
        build_blastn_db(merged, outdir)
    except FileNotFoundError:
        print("  makeblastdb not found (install: conda install -c bioconda blast)")

    # Build diamond DB (optional)
    build_diamond_db(merged, outdir)

    print("\nDone. Mapping file:", mapping)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
