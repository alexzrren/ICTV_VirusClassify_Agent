"""
HMMER wrapper for extracting target genomic regions from query sequences.

Supports family-specific HMM profiles (built by scripts/build_family_hmms.py)
as well as legacy RdRp profiles.  Runtime flow:

  1. Resolve HMM path: data/hmm/{Family}_targets.hmm (preferred) or legacy dir
  2. getorf to predict ORFs from nucleotide query
  3. hmmsearch each ORF against the family HMM
  4. Return extracted protein region(s)
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_HMM_DIR = _PROJECT_ROOT / "data" / "hmm"
_HMM_TARGETS = _PROJECT_ROOT / "data" / "hmm_targets.json"
_LEGACY_HMM_DIR = os.environ.get(
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
            ["/home/renzirui/micromamba/bin/hmmsearch",
             "--domtblout", domtbl_path, "-E", "1e-3",
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


def _resolve_hmm_path(family: str) -> Optional[Path]:
    """Find the HMM profile for a family. Checks family-specific targets first."""
    # 1. Family-specific combined HMM (from build_family_hmms.py)
    combined = _HMM_DIR / f"{family}_targets.hmm"
    if combined.exists():
        return combined
    # 2. Coronaviridae special case
    cov = _HMM_DIR / "CoV_5domains.hmm"
    if family.lower() == "coronaviridae" and cov.exists():
        return cov
    # 3. Legacy RdRp profiles
    legacy = Path(_LEGACY_HMM_DIR)
    for pattern in [f"{family}.hmm", f"{family.lower()}.hmm"]:
        matches = list(legacy.glob(pattern))
        if matches:
            return matches[0]
    return None


def list_available_hmms() -> dict[str, list[str]]:
    """Return {family: [region_names]} for all families with HMM profiles."""
    result = {}
    for hmm_file in _HMM_DIR.glob("*_targets.hmm"):
        family = hmm_file.name.replace("_targets.hmm", "")
        # Read HMM names from the file
        regions = []
        for line in hmm_file.open():
            if line.startswith("NAME"):
                name = line.split()[1].replace(f"{family}_", "")
                regions.append(name)
        if regions:
            result[family] = regions
    # Coronaviridae special
    cov = _HMM_DIR / "CoV_5domains.hmm"
    if cov.exists() and "Coronaviridae" not in result:
        result["Coronaviridae"] = ["3CLpro", "NiRAN", "RdRp", "ZBD", "HEL1"]
    return result


def _run_getorf(
    nucleotide_seq: str, min_size: int = 300
) -> dict[str, str]:
    """Run EMBOSS getorf and return {short_id: protein_seq}."""
    result = _run_getorf_with_headers(nucleotide_seq, min_size)
    return result[0]


def _run_getorf_with_headers(
    nucleotide_seq: str, min_size: int = 300
) -> tuple[dict[str, str], dict[str, str]]:
    """Run EMBOSS getorf. Returns (seqs, full_headers).

    seqs: {short_id: protein_seq}
    full_headers: {short_id: full_header_line} — includes '[start - end]' coords.
    """
    getorf_bin = "/home/renzirui/micromamba/bin/_getorf"
    env = os.environ.copy()
    emboss_bin = str(Path(getorf_bin).parent)
    env["EMBOSS_ACDROOT"] = str(Path(emboss_bin) / ".." / "share" / "EMBOSS" / "acd")
    env["EMBOSS_DATA"] = str(Path(emboss_bin) / ".." / "share" / "EMBOSS" / "data")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".fasta", delete=False) as f:
        f.write(f">query\n{nucleotide_seq}\n")
        nuc_path = f.name
    orf_path = nuc_path + ".orf"
    try:
        subprocess.run(
            [getorf_bin, "-sequence", nuc_path, "-outseq", orf_path,
             "-find", "1", "-minsize", str(min_size)],
            capture_output=True, text=True, check=False, env=env,
        )
        if not Path(orf_path).exists():
            return {}, {}
        raw = Path(orf_path).read_text()
        return _parse_fasta_seqs(raw), _parse_fasta_full_headers(raw)
    finally:
        Path(nuc_path).unlink(missing_ok=True)
        Path(orf_path).unlink(missing_ok=True)


def _parse_orf_coords(header: str) -> tuple[int, int, bool]:
    """Parse getorf header like 'seq_1 [123 - 456]' → (start, end, is_reverse).

    Returns 1-based inclusive (start, end) from the header.
    If start > end, the ORF is on the reverse strand.
    """
    m = re.search(r'\[(\d+)\s*-\s*(\d+)\]', header)
    if not m:
        return (0, 0, False)
    s, e = int(m.group(1)), int(m.group(2))
    return (s, e, s > e)


def extract_hmm_region(nucleotide_seq: str, family: str,
                        hmm_dir: Optional[str] = None) -> Optional[str]:
    """
    Extract the best HMM-matched protein region from a nucleotide sequence.
    Returns the single best-matching protein subsequence, or None.
    """
    hmm_path = _resolve_hmm_path(family)
    if not hmm_path:
        raise FileNotFoundError(f"HMM profile not found for {family}")

    orfs = _run_getorf(nucleotide_seq)
    if not orfs:
        return None

    best_protein = None
    best_score = 0.0
    for header, prot_seq in orfs.items():
        hits = hmmsearch_protein(prot_seq, str(hmm_path))
        if hits and hits[0]["score"] > best_score:
            best_score = hits[0]["score"]
            h = hits[0]
            best_protein = prot_seq[h["start"]:h["end"]]

    return best_protein


class RegionResult:
    """Holds both protein and nucleotide sequences for an extracted region."""
    __slots__ = ("protein", "nucleotide")

    def __init__(self, protein: str, nucleotide: str = ""):
        self.protein = protein
        self.nucleotide = nucleotide


def extract_all_regions(
    nucleotide_seq: str, family: str
) -> dict[str, str]:
    """Extract HMM-defined target protein regions from a nucleotide sequence.

    Returns {region_name: protein_sequence} for backward compatibility.
    Use extract_all_regions_with_nt() to also get nucleotide subsequences.
    """
    results = extract_all_regions_with_nt(nucleotide_seq, family)
    return {k: v.protein for k, v in results.items()}


def extract_all_regions_with_nt(
    nucleotide_seq: str, family: str
) -> dict[str, RegionResult]:
    """Extract HMM-defined target protein regions AND their nucleotide subsequences.

    For each region, returns a RegionResult with both the protein sequence
    (from the HMM domain hit) and the corresponding nucleotide subsequence
    (mapped back via getorf ORF coordinates + alignment boundaries).
    """
    hmm_path = _resolve_hmm_path(family)
    if not hmm_path:
        return {}

    orfs, orf_headers = _run_getorf_with_headers(nucleotide_seq, min_size=150)
    if not orfs:
        return {}

    # region -> (score, protein, nt_subseq)
    best_per_region: dict[str, tuple[float, str, str]] = {}

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        orf_fa = tmp / "orfs.faa"
        with open(orf_fa, "w") as f:
            for hdr, seq in orfs.items():
                f.write(f">{hdr}\n{seq}\n")

        hmmsearch_bin = "/home/renzirui/micromamba/bin/hmmsearch"
        dom_out = tmp / "dom.tbl"
        subprocess.run(
            [hmmsearch_bin, "--noali", "--domtblout", str(dom_out),
             "-E", "1e-5", str(hmm_path), str(orf_fa)],
            capture_output=True, text=True,
        )

        if not dom_out.exists():
            return {}

        for line in dom_out.read_text().splitlines():
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 22:
                continue
            target_name = parts[0]  # ORF short_id (e.g. 'query_1')
            hmm_name = parts[3]     # e.g. Paramyxoviridae_L_protein
            score = float(parts[13])
            ali_start = int(parts[17]) - 1  # 0-based protein start
            ali_end = int(parts[18])        # exclusive protein end

            # Strip family prefix to get region name
            region = hmm_name
            prefix = f"{family}_"
            if hmm_name.startswith(prefix):
                region = hmm_name[len(prefix):]

            prot_seq = orfs.get(target_name, "")
            extracted_prot = prot_seq[ali_start:ali_end]

            if len(extracted_prot) <= 50:
                continue
            if region in best_per_region and score <= best_per_region[region][0]:
                continue

            # Map protein domain boundaries back to nucleotide coordinates
            # using the full getorf header which contains [nt_start - nt_end].
            nt_subseq = ""
            full_hdr = orf_headers.get(target_name, "")
            if full_hdr:
                orf_start, orf_end, is_rev = _parse_orf_coords(full_hdr)
                if orf_start > 0:
                    if is_rev:
                        # Reverse-strand ORF: orf_start > orf_end (1-based)
                        nt_from = orf_start - ali_start * 3
                        nt_to = orf_start - ali_end * 3
                        lo = min(nt_from, nt_to) - 1  # 0-based
                        hi = max(nt_from, nt_to)
                        chunk = nucleotide_seq[lo:hi]
                        comp = str.maketrans("ACGTacgt", "TGCAtgca")
                        nt_subseq = chunk[::-1].translate(comp)
                    else:
                        # Forward-strand ORF: orf_start < orf_end
                        nt_from = orf_start - 1 + ali_start * 3  # 0-based
                        nt_to = orf_start - 1 + ali_end * 3
                        nt_subseq = nucleotide_seq[nt_from:nt_to]

            best_per_region[region] = (score, extracted_prot, nt_subseq)

    return {
        k: RegionResult(protein=v[1], nucleotide=v[2])
        for k, v in best_per_region.items()
    }


def _parse_fasta_seqs(text: str) -> dict[str, str]:
    """Parse FASTA text → {short_id: sequence}.

    Uses the first token of the header as key (compatible with hmmsearch
    domtblout target_name field).
    """
    seqs: dict[str, str] = {}
    cur = ""
    for line in text.splitlines():
        if line.startswith(">"):
            cur = line[1:].split()[0]
            seqs[cur] = ""
        elif cur:
            seqs[cur] += line.strip().replace("*", "")
    return seqs


def _parse_fasta_full_headers(text: str) -> dict[str, str]:
    """Parse FASTA text → {short_id: full_header_line}.

    Builds a lookup from short ID (e.g. 'query_1') to the full header
    including coordinate annotations like '[123 - 456]'.
    """
    headers: dict[str, str] = {}
    for line in text.splitlines():
        if line.startswith(">"):
            full = line[1:].strip()
            short = full.split()[0]
            headers[short] = full
    return headers
