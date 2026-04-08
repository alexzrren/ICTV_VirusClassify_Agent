#!/usr/bin/env python3
"""
Build HMM profiles for 5 Coronaviridae replicase domains from reference sequences.

Domain coordinates (aa) in SARS-CoV-2 ORF1ab polyprotein (MN908947.3):
  ORF1ab encodes pp1ab (~7096 aa after ribosomal slippage).
  Nsp numbering relative to ORF1ab:
    3CLpro   = Nsp5:   aa 3264-3569   (306 aa)
    NiRAN    = Nsp12:  aa 4393-4535   (143 aa)  [N-terminal NiRAN domain]
    RdRp     = Nsp12:  aa 4536-4932   (397 aa)  [RdRp palm/fingers/thumb]
    ZBD      = Nsp13:  aa 5316-5443   (128 aa)  [Zinc-binding domain]
    HEL1     = Nsp13:  aa 5444-5836   (393 aa)  [Helicase RecA-like domain]

These coordinates follow Ziebuhr et al. 2018 / Gorbalenya et al. 2021.
"""

from __future__ import annotations
import os, sys, subprocess, textwrap
from pathlib import Path
from Bio import SeqIO
from Bio.Seq import Seq

HMMER = Path("/home/renzirui/micromamba/bin")
REFDIR = Path("/home/renzirui/Projects/ICTV/ictv_classifier/reference/Coronaviridae")
HMMDIR = Path("/home/renzirui/Projects/ICTV/ictv_agent/data/hmm")
MAFFT = "/home/renzirui/micromamba/bin/mafft"

# Domain definitions: (name, orf1ab_aa_start, orf1ab_aa_end) — 1-based inclusive
# These are for SARS-CoV-2 (MN908947.3) ORF1ab pp1ab coordinates
DOMAINS = {
    "3CLpro":  (3264, 3569),   # Nsp5
    "NiRAN":   (4393, 4535),   # Nsp12 N-terminal
    "RdRp":    (4536, 4932),   # Nsp12 catalytic core
    "ZBD":     (5316, 5443),   # Nsp13 zinc-binding
    "HEL1":    (5444, 5836),   # Nsp13 helicase
}

# ORF1ab start/end in SARS-CoV-2 MN908947.3 (nt, 1-based)
# ORF1a: 266-13468, ORF1b starts at 13468 with -1 frameshift
# Full pp1ab: nt 266..21555 (ribosomal slippage at 13465)
ORF1AB_NT_START = 266    # nt 1-based
ORF1AB_NT_END   = 21555

SARSCOV2_ACC = "MN908947.3"


def find_sarscov2(fasta: Path) -> SeqIO.SeqRecord | None:
    for rec in SeqIO.parse(str(fasta), "fasta"):
        if SARSCOV2_ACC in rec.id or "MN908947" in rec.id:
            return rec
        if "Wuhan-Hu-1" in rec.description or "SARS coronavirus 2" in rec.description.lower():
            return rec
    return None


def translate_orf1ab(genome_seq: Seq) -> Seq | None:
    """Translate ORF1ab with automatic frameshift site detection.
    Delegates to corona_pud.translate_orf1ab which handles all Orthocoronavirinae."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from backend.tools.corona_pud import translate_orf1ab as _translate
    result = _translate(str(genome_seq))
    if result is None:
        return None
    return Seq(result)


def extract_domain(pp1ab: Seq, start: int, end: int) -> Seq:
    """Extract domain from pp1ab (1-based coords, inclusive)."""
    return pp1ab[start - 1: end]


def get_orf1ab_for_seq(rec: SeqIO.SeqRecord) -> Seq | None:
    """Translate ORF1ab from a coronavirus genome record."""
    return translate_orf1ab(rec.seq.upper())


def build_domain_seeds(fasta: Path, outdir: Path) -> dict[str, Path]:
    """Extract domain sequences from reference genomes → seed MSAs."""
    outdir.mkdir(parents=True, exist_ok=True)
    domain_seqs: dict[str, list[tuple[str, Seq]]] = {d: [] for d in DOMAINS}

    print("Translating reference ORF1ab sequences...")
    records = list(SeqIO.parse(str(fasta), "fasta"))

    # First pass: get SARS-CoV-2 to establish coordinate reference
    sarscov2 = find_sarscov2(fasta)
    if sarscov2 is None:
        print("WARNING: SARS-CoV-2 not found, using first record as reference")
        sarscov2 = records[0]

    ref_pp1ab = translate_orf1ab(sarscov2.seq.upper())
    print(f"  SARS-CoV-2 pp1ab: {len(ref_pp1ab)} aa")

    # Extract domains from all reference sequences using coordinate scaling
    SARS2_LEN = len(ref_pp1ab)  # ~7096 aa
    for rec in records:
        if len(rec.seq) < 20000:
            continue
        try:
            pp1ab = translate_orf1ab(rec.seq.upper())
            if pp1ab is None or len(pp1ab) < 3000:
                continue
            scale = len(pp1ab) / SARS2_LEN
            for dom_name, (start, end) in DOMAINS.items():
                s = max(0, int((start - 1) * scale))
                e = min(len(pp1ab), int(end * scale))
                if e > s + 50:
                    dom_seq = pp1ab[s: e]
                    safe_id = rec.id.replace(".", "_").replace(" ", "_")[:30]
                    domain_seqs[dom_name].append((safe_id, dom_seq))
        except Exception as e:
            pass

    # Write seed FASTAs
    seed_files = {}
    for dom_name, seqs in domain_seqs.items():
        if not seqs:
            print(f"  WARNING: no sequences extracted for {dom_name}")
            continue
        seed_fa = outdir / f"{dom_name}_seeds.faa"
        with open(seed_fa, "w") as f:
            for sid, seq in seqs:
                f.write(f">{sid}\n{textwrap.fill(str(seq), 60)}\n")
        print(f"  {dom_name}: {len(seqs)} seed sequences → {seed_fa.name}")
        seed_files[dom_name] = seed_fa

    return seed_files


def align_seeds(seed_files: dict[str, Path], outdir: Path) -> dict[str, Path]:
    """Align seed sequences with MAFFT."""
    aln_files = {}
    for dom_name, seed_fa in seed_files.items():
        aln_fa = outdir / f"{dom_name}_aln.faa"
        print(f"  Aligning {dom_name}...")
        cmd = [MAFFT, "--auto", "--quiet", str(seed_fa)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"    MAFFT failed: {result.stderr[:100]}")
            continue
        aln_fa.write_text(result.stdout)
        aln_files[dom_name] = aln_fa
    return aln_files


def build_hmms(aln_files: dict[str, Path], hmmdir: Path) -> dict[str, Path]:
    """Build HMM profiles with hmmbuild."""
    hmmdir.mkdir(parents=True, exist_ok=True)
    hmm_files = {}
    for dom_name, aln_fa in aln_files.items():
        hmm_out = hmmdir / f"CoV_{dom_name}.hmm"
        print(f"  Building HMM for {dom_name}...")
        cmd = [str(HMMER / "hmmbuild"), "--amino", "-n", f"CoV_{dom_name}",
               str(hmm_out), str(aln_fa)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"    hmmbuild failed: {result.stderr[:200]}")
            continue
        hmm_files[dom_name] = hmm_out
        print(f"    → {hmm_out.name}")
    return hmm_files


def press_hmms(hmmdir: Path) -> Path:
    """Concatenate and press all HMMs into a searchable database."""
    combined = hmmdir / "CoV_5domains.hmm"
    hmm_files = sorted(hmmdir.glob("CoV_*.hmm"))
    hmm_files = [f for f in hmm_files if "5domains" not in f.name]

    # Concatenate
    with open(combined, "w") as out:
        for hmm in hmm_files:
            out.write(hmm.read_text())

    # Press
    cmd = [str(HMMER / "hmmpress"), "-f", str(combined)]
    subprocess.run(cmd, capture_output=True)
    print(f"  Combined HMM: {combined}")
    return combined


def main():
    fasta = REFDIR / "sequences.fasta"
    if not fasta.exists():
        print(f"ERROR: {fasta} not found")
        sys.exit(1)

    tmpdir = HMMDIR / "seeds"

    print("=== Step 1: Extract domain seeds from reference sequences ===")
    seed_files = build_domain_seeds(fasta, tmpdir)

    print("\n=== Step 2: Align seeds with MAFFT ===")
    aln_files = align_seeds(seed_files, tmpdir)

    print("\n=== Step 3: Build HMM profiles with hmmbuild ===")
    hmm_files = build_hmms(aln_files, HMMDIR)

    print("\n=== Step 4: Press HMMs for hmmsearch ===")
    combined = press_hmms(HMMDIR)

    print(f"\nDone. HMMs: {list(hmm_files.keys())}")
    print(f"Combined: {combined}")


if __name__ == "__main__":
    main()
