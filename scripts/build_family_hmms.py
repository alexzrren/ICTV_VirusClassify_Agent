#!/usr/bin/env python3
"""
Build HMM profiles for all virus families defined in data/hmm_targets.json.

For each family/region:
  1. Pick a seed sequence (longest ORF matching min_orf_aa from first reference)
  2. tblastn seed against all reference genomes to extract homologous proteins
  3. MAFFT multiple sequence alignment
  4. hmmbuild → HMM profile
  5. hmmpress → searchable database

Usage:
    python scripts/build_family_hmms.py                          # all families
    python scripts/build_family_hmms.py --family Paramyxoviridae # single family
    python scripts/build_family_hmms.py --dry-run                # preview only
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

from Bio import SeqIO
from Bio.Seq import Seq

# ── Paths ────────────────────────────────────────────────────────────────

PROJECT  = Path(__file__).parent.parent
HMMER    = Path(os.environ.get("HMMER_BIN", "/home/renzirui/micromamba/bin"))
MAFFT    = os.environ.get("MAFFT_BIN", "/home/renzirui/micromamba/bin/mafft")
GETORF   = os.environ.get("GETORF_BIN", "/home/renzirui/micromamba/bin/_getorf")
TBLASTN  = os.environ.get("TBLASTN_BIN", str(Path("/home/renzirui/micromamba/bin/tblastn")))
MAKEBLASTDB = os.environ.get("MAKEBLASTDB_BIN", str(Path("/home/renzirui/micromamba/bin/makeblastdb")))

TARGETS  = PROJECT / "data" / "hmm_targets.json"
REF_DIR  = PROJECT / "data" / "references"
HMM_DIR  = PROJECT / "data" / "hmm"
SEED_DIR = HMM_DIR / "seeds"

# How many reference sequences to include at most (avoids very slow builds)
MAX_SEEDS = 80


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a command with proxy env vars removed and EMBOSS env set."""
    env = os.environ.copy()
    for k in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
              "all_proxy", "ALL_PROXY"]:
        env.pop(k, None)
    # EMBOSS environment for _getorf
    emboss_bin = str(Path(GETORF).parent)
    env["EMBOSS_ACDROOT"] = str(Path(emboss_bin) / ".." / "share" / "EMBOSS" / "acd")
    env["EMBOSS_DATA"] = str(Path(emboss_bin) / ".." / "share" / "EMBOSS" / "data")
    return subprocess.run(cmd, env=env, **kwargs)


# ── Step 1: Seed extraction ─────────────────────────────────────────────

def get_orfs(nucleotide_seq: str, min_aa: int = 100) -> list[tuple[str, str]]:
    """Predict ORFs with EMBOSS getorf. Returns [(header, protein_seq), ...]."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".fna", delete=False) as f:
        f.write(f">seq\n{textwrap.fill(nucleotide_seq, 60)}\n")
        nuc_path = f.name
    orf_path = nuc_path + ".orf"
    try:
        r = _run([GETORF, "-sequence", nuc_path, "-outseq", orf_path,
                  "-find", "1", "-minsize", str(min_aa * 3)],
                 capture_output=True, text=True)
        if r.returncode != 0 or not Path(orf_path).exists():
            return []
        orfs = []
        cur_hdr, cur_seq = "", ""
        for line in open(orf_path):
            if line.startswith(">"):
                if cur_hdr and cur_seq:
                    orfs.append((cur_hdr, cur_seq.replace("*", "")))
                cur_hdr = line.strip()
                cur_seq = ""
            else:
                cur_seq += line.strip()
        if cur_hdr and cur_seq:
            orfs.append((cur_hdr, cur_seq.replace("*", "")))
        return orfs
    finally:
        Path(nuc_path).unlink(missing_ok=True)
        Path(orf_path).unlink(missing_ok=True)


def pick_seed(fasta_path: Path, min_orf_aa: int) -> tuple[str, str] | None:
    """Pick the longest ORF >= min_orf_aa from the first few reference sequences."""
    best = None
    best_len = 0
    for i, rec in enumerate(SeqIO.parse(str(fasta_path), "fasta")):
        if i >= 5:  # check first 5 sequences
            break
        orfs = get_orfs(str(rec.seq), min_orf_aa)
        for hdr, seq in orfs:
            if len(seq) >= min_orf_aa and len(seq) > best_len:
                best = (rec.id, seq)
                best_len = len(seq)
    return best


# ── Step 2: tblastn extraction ───────────────────────────────────────────

def tblastn_extract(seed_protein: str, fasta_path: Path,
                    min_orf_aa: int, max_seqs: int = MAX_SEEDS) -> list[tuple[str, str]]:
    """
    Use tblastn to find homologs of seed_protein in all reference genomes.
    Returns [(accession, protein_seq), ...].
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # Write seed protein
        seed_fa = tmp / "seed.faa"
        seed_fa.write_text(f">seed\n{textwrap.fill(seed_protein, 60)}\n")

        # Build temporary BLAST nucleotide DB
        db_prefix = tmp / "refdb"
        r = _run([MAKEBLASTDB, "-in", str(fasta_path), "-dbtype", "nucl",
                  "-out", str(db_prefix)], capture_output=True, text=True)
        if r.returncode != 0:
            print(f"    makeblastdb failed: {r.stderr[:200]}")
            return []

        # Run tblastn
        out_file = tmp / "tblastn.tsv"
        r = _run([
            TBLASTN, "-query", str(seed_fa), "-db", str(db_prefix),
            "-outfmt", "6 sseqid sstart send sframe slen evalue bitscore",
            "-evalue", "1e-10", "-max_target_seqs", str(max_seqs * 2),
            "-num_threads", "4", "-out", str(out_file),
        ], capture_output=True, text=True)
        if r.returncode != 0:
            print(f"    tblastn failed: {r.stderr[:200]}")
            return []

        # Parse hits - keep best hit per subject
        best_hits: dict[str, tuple[int, int, int, float]] = {}  # acc -> (start, end, frame, bitscore)
        for line in out_file.read_text().splitlines():
            parts = line.strip().split("\t")
            if len(parts) < 7:
                continue
            acc = parts[0]
            sstart, send = int(parts[1]), int(parts[2])
            frame = int(parts[3])
            bitscore = float(parts[6])
            if acc not in best_hits or bitscore > best_hits[acc][3]:
                best_hits[acc] = (sstart, send, frame, bitscore)

        # Extract protein from each hit
        # Build a quick lookup of reference sequences
        ref_seqs = {}
        for rec in SeqIO.parse(str(fasta_path), "fasta"):
            ref_seqs[rec.id] = str(rec.seq)

        results = []
        for acc, (sstart, send, frame, bitscore) in best_hits.items():
            if acc not in ref_seqs:
                continue
            genome = ref_seqs[acc]
            # Extract the matched region and translate
            if sstart <= send:
                subseq = genome[sstart - 1: send]
            else:
                subseq = str(Seq(genome[send - 1: sstart]).reverse_complement())

            try:
                protein = str(Seq(subseq).translate(to_stop=False)).replace("*", "X").rstrip("X")
            except Exception:
                continue

            if len(protein) >= min_orf_aa:
                # Trim to longest stop-free region
                parts = protein.split("X")
                longest = max(parts, key=len) if parts else protein
                if len(longest) >= min_orf_aa:
                    safe_id = acc.replace(".", "_")[:30]
                    results.append((safe_id, longest))

        return results[:max_seqs]


# ── Step 3-4: Align and build HMM ────────────────────────────────────────

def align_sequences(seqs: list[tuple[str, str]], out_aln: Path) -> bool:
    """MAFFT alignment."""
    in_fa = out_aln.with_suffix(".seeds.faa")
    with open(in_fa, "w") as f:
        for sid, seq in seqs:
            f.write(f">{sid}\n{textwrap.fill(seq, 60)}\n")

    r = _run([MAFFT, "--auto", "--quiet", "--thread", "4", str(in_fa)],
             capture_output=True, text=True)
    if r.returncode != 0:
        print(f"    MAFFT failed: {r.stderr[:200]}")
        return False
    out_aln.write_text(r.stdout)
    return True


def build_hmm(aln_path: Path, hmm_path: Path, name: str) -> bool:
    """hmmbuild from alignment."""
    r = _run([str(HMMER / "hmmbuild"), "--amino", "-n", name,
              str(hmm_path), str(aln_path)],
             capture_output=True, text=True)
    if r.returncode != 0:
        print(f"    hmmbuild failed: {r.stderr[:200]}")
        return False
    return True


def press_combined(family: str, region_hmms: list[Path], combined_path: Path) -> bool:
    """Concatenate region HMMs and hmmpress."""
    with open(combined_path, "w") as out:
        for hmm in region_hmms:
            out.write(hmm.read_text())

    r = _run([str(HMMER / "hmmpress"), "-f", str(combined_path)],
             capture_output=True, text=True)
    if r.returncode != 0:
        print(f"    hmmpress failed: {r.stderr[:200]}")
        return False
    return True


# ── Main pipeline ─────────────────────────────────────────────────────────

def build_family(family: str, config: dict, dry_run: bool = False) -> bool:
    """Build HMM profiles for one family."""
    regions = config.get("regions", [])
    if not regions:
        return True

    fasta_path = REF_DIR / family / "sequences.fasta"
    if not fasta_path.exists():
        print(f"  WARNING: {fasta_path} not found, skipping")
        return False

    nref = sum(1 for _ in SeqIO.parse(str(fasta_path), "fasta"))
    print(f"\n{'='*60}")
    print(f"  {family}: {nref} references, {len(regions)} region(s)")

    if dry_run:
        for reg in regions:
            print(f"    - {reg['name']} (min_orf_aa={reg['min_orf_aa']})")
        return True

    seed_dir = SEED_DIR / family
    seed_dir.mkdir(parents=True, exist_ok=True)

    region_hmms = []
    for reg in regions:
        rname = reg["name"]
        min_aa = reg["min_orf_aa"]
        hmm_name = f"{family}_{rname}"
        print(f"\n  --- {rname} (min {min_aa} aa) ---")

        # Step 1: Pick seed
        print(f"    Picking seed protein...")
        seed = pick_seed(fasta_path, min_aa)
        if not seed:
            print(f"    WARNING: No seed found for {rname}, skipping")
            continue
        seed_acc, seed_prot = seed
        print(f"    Seed: {seed_acc} ({len(seed_prot)} aa)")

        # Step 2: tblastn to find homologs
        print(f"    tblastn extraction from {nref} references...")
        homologs = tblastn_extract(seed_prot, fasta_path, min_aa)
        if not homologs:
            print(f"    WARNING: No homologs found, skipping")
            continue
        print(f"    Extracted {len(homologs)} homologous sequences")

        # Step 3: MAFFT alignment
        aln_path = seed_dir / f"{rname}_aln.faa"
        print(f"    MAFFT alignment...")
        if not align_sequences(homologs, aln_path):
            continue

        # Step 4: hmmbuild
        hmm_path = HMM_DIR / f"{hmm_name}.hmm"
        print(f"    hmmbuild → {hmm_path.name}")
        if not build_hmm(aln_path, hmm_path, hmm_name):
            continue
        region_hmms.append(hmm_path)

    # Step 5: Combine and press
    if region_hmms:
        combined = HMM_DIR / f"{family}_targets.hmm"
        print(f"\n  Combining {len(region_hmms)} HMMs → {combined.name}")
        if press_combined(family, region_hmms, combined):
            print(f"  ✓ {family}: {len(region_hmms)} HMM profile(s) built")
            return True

    return False


def main():
    ap = argparse.ArgumentParser(description="Build HMM profiles for ICTV virus families.")
    ap.add_argument("--family", help="Build for a single family only")
    ap.add_argument("--dry-run", action="store_true", help="Preview what would be built")
    args = ap.parse_args()

    if not TARGETS.exists():
        print(f"ERROR: {TARGETS} not found", file=sys.stderr)
        sys.exit(1)

    with open(TARGETS) as f:
        targets = json.load(f)

    # Remove comment keys
    targets = {k: v for k, v in targets.items() if not k.startswith("_")}

    HMM_DIR.mkdir(parents=True, exist_ok=True)
    SEED_DIR.mkdir(parents=True, exist_ok=True)

    families = [args.family] if args.family else sorted(targets.keys())
    success = 0
    skipped = 0

    for family in families:
        if family not in targets:
            print(f"WARNING: {family} not in hmm_targets.json")
            continue
        config = targets[family]
        if not config.get("regions"):
            skipped += 1
            continue
        if build_family(family, config, dry_run=args.dry_run):
            success += 1

    print(f"\n{'='*60}")
    print(f"Done. Built: {success}, Skipped: {skipped}, Total: {len(families)}")


if __name__ == "__main__":
    main()
