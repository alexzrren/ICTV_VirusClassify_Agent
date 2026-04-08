"""
Coronaviridae subgenus/genus/subfamily/species classification via DEmARC PUD.

Workflow:
1. Translate ORF1ab from query genome (nucleotide input)
2. Use hmmsearch to extract 5 replicase domains (3CLpro, NiRAN, RdRp, ZBD, HEL1)
3. Align domains against reference set via hmmalign
4. Compute pairwise uncorrected distance (PUD) to each reference
5. Map to taxonomy via DEmARC thresholds

PUD thresholds (Coronaviridae, Table 4, Ziebuhr et al. 2021):
  Species:   PUD  7.5%  (PPD 0.095)
  Subgenus:  PUD 13.2-14.2%  (PPD 0.200-0.221)
  Genus:     PUD 35.1-36.0%  (PPD 0.873-0.909)
  Subfamily: PUD 46.8-51.0%  (PPD 1.472-1.757)
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import textwrap
from pathlib import Path
from typing import Optional

from Bio import SeqIO
from Bio.Seq import Seq

HMMER_BIN = Path("/home/renzirui/micromamba/bin")
MAFFT_BIN = "/home/renzirui/micromamba/bin/mafft"
HMM_FILE  = Path(__file__).parent.parent.parent / "data" / "hmm" / "CoV_5domains.hmm"
REF_FASTA = Path("/home/renzirui/Projects/ICTV/ictv_classifier/reference/Coronaviridae/sequences.fasta")
TAX_DB    = Path(__file__).parent.parent.parent / "data" / "taxonomy.db"

# DEmARC PUD thresholds
THRESHOLDS = {
    "species":   0.075,
    "subgenus":  0.142,   # upper bound of subgenus range
    "genus":     0.360,   # upper bound of genus range
    "subfamily": 0.510,   # upper bound of subfamily range
    "family":    0.681,
}

# Conserved slippery sequence for -1 ribosomal frameshift in Orthocoronavirinae
# The UUUAAAC heptamer is at the 3' end of ORF1a
SLIPPERY_SEQ = "TTTAAAC"
# Conserved stem-loop immediately downstream to confirm frameshift site
SLIPPERY_DOWNSTREAM = "TTTAAACGAAATTTG"  # partial pattern

# Domain coordinates in pp1ab (1-based aa, relative to SARS-CoV-2 concatenated ORF1a+ORF1b)
# Used as fallback if hmmsearch fails
DOMAIN_COORDS_SARS2 = {
    "3CLpro": (3264, 3569),
    "NiRAN":  (4393, 4535),
    "RdRp":   (4536, 4932),
    "ZBD":    (5316, 5443),
    "HEL1":   (5444, 5836),
}
DOMAIN_ORDER = ["3CLpro", "NiRAN", "RdRp", "ZBD", "HEL1"]

# Expected domain sizes (aa) — used to reject oversized HMM hits
DOMAIN_EXPECTED_SIZE = {k: (end - start + 1) for k, (start, end) in DOMAIN_COORDS_SARS2.items()}
# Allow up to 2x expected size to accommodate divergent viruses
DOMAIN_SIZE_TOLERANCE = 2.0


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    for k in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
              "all_proxy", "ALL_PROXY"]:
        env.pop(k, None)
    return subprocess.run(cmd, env=env, **kwargs)


def orient_genome(genome: str) -> tuple[str, int]:
    """
    Determine genome orientation by ORF analysis.

    Scans all 6 reading frames (3 forward + 3 reverse complement) for the
    longest stop-codon-free region.  Coronavirus ORF1a is always the longest
    ORF on the genome (>4 000 aa), so the frame that contains it unambiguously
    identifies the correct strand and reading frame.

    Returns (oriented_genome, longest_orf_nt_start_0based).
    ``longest_orf_nt_start`` is the nucleotide offset (in the oriented genome)
    of the first codon in the longest stop-free region — useful as an upper
    bound when searching for the ORF1a ATG.
    """
    best_genome = genome
    best_orf_nt_start = 0
    best_orf_len = 0
    rc = str(Seq(genome).reverse_complement())

    for seq in (genome, rc):
        for frame in range(3):
            prot = str(Seq(seq[frame:]).translate())
            orfs = prot.split("*")
            preceding_aa = 0
            for orf in orfs:
                if len(orf) > best_orf_len:
                    best_orf_len = len(orf)
                    best_genome = seq
                    best_orf_nt_start = frame + preceding_aa * 3
                preceding_aa += len(orf) + 1  # +1 for the stop codon

    return best_genome, best_orf_nt_start


def find_orf1a_start(genome: str, longest_orf_start: int = 0) -> int:
    """
    Find ORF1a start codon.

    Searches for the first ATG upstream of (or at) the longest ORF region
    that initiates an ORF > 3 000 aa.  ``longest_orf_start`` is a hint from
    ``orient_genome`` — we search backwards from there and also a small
    window forward, so truncated 5'-contigs are handled.

    Returns 1-based nucleotide position.
    """
    # Search window: from a bit before the longest-ORF start up to 1000 nt
    # past it (the ATG may be slightly inside the region).
    search_from = max(0, longest_orf_start - 500)
    search_to = min(longest_orf_start + 1000, len(genome) - 2)
    for i in range(search_from, search_to):
        if genome[i:i+3] == "ATG":
            test = Seq(genome[i: i + 15000]).translate(to_stop=True)
            if len(test) > 3000:
                return i + 1  # 1-based
    # Fallback: use the start of the longest ORF directly (contig truncated
    # at 5' — no ATG visible, but the reading frame is correct).
    if longest_orf_start > 0:
        return longest_orf_start + 1
    return 266  # last resort: SARS-CoV-2 known start


def find_frameshift_site(genome: str, orf1a_start: int) -> int:
    """
    Find the -1 ribosomal frameshift site by locating the conserved
    slippery sequence TTTAAAC within ORF1a.
    Returns the 0-based nt position of the start of the slippery sequence.

    Validates each candidate by translating the resulting ORF1b and checking
    that it encodes a protein of expected length (>1500 aa). This avoids
    false-positive matches for divergent coronaviruses (e.g. SARS-CoV-1)
    where the real frameshift site is at a different position than in SARS-CoV-2.
    """
    # Wide search window: slippage is in the second half of ORF1a.
    # Start at +8000 (not +10000) to catch divergent genera.
    search_start = orf1a_start + 8000
    search_end   = min(orf1a_start + 20000, len(genome))
    region = genome[search_start: search_end]

    # Collect ALL candidate positions for TTTAAAC in the window
    candidates: list[int] = []
    offset = 0
    while True:
        p = region.find(SLIPPERY_SEQ, offset)
        if p < 0:
            break
        candidates.append(search_start + p)
        offset = p + 1

    # Validate each candidate: the real frameshift site produces an ORF1b of
    # ~2700 aa (Orthocoronavirinae). Accept any that yields >1500 aa; pick
    # the one with the longest resulting ORF1b.
    best_pos = -1
    best_len = 0
    for slip_pos in candidates:
        orf1b_start = slip_pos + len(SLIPPERY_SEQ) - 1  # -1 slip
        orf1b_end   = min(orf1b_start + 12000, len(genome))
        try:
            pp1b_test = str(Seq(genome[orf1b_start: orf1b_end]).translate(to_stop=True))
        except Exception:
            continue
        if len(pp1b_test) > best_len and len(pp1b_test) > 1500:
            best_len = len(pp1b_test)
            best_pos = slip_pos

    if best_pos >= 0:
        return best_pos

    # Broader fallback (no ORF1b validation): search 5000–22000 nt
    broad = genome[5000: min(22000, len(genome))]
    p = broad.find(SLIPPERY_SEQ)
    if p >= 0:
        return 5000 + p
    return -1


def translate_orf1ab(genome_seq: str) -> Optional[str]:
    """
    Translate ORF1ab from genome nucleotide sequence.

    1. ``orient_genome`` scans all 6 reading frames to find the strand
       carrying the longest ORF (= ORF1a for any coronavirus).
    2. ``find_orf1a_start`` locates the ATG near that ORF.
    3. ``find_frameshift_site`` finds the TTTAAAC slippery sequence.
    4. ORF1a + ORF1b are translated and concatenated into pp1ab.

    Returns the amino-acid string, or *None* on failure.
    """
    genome = genome_seq.upper().replace(" ", "").replace("\n", "")
    if len(genome) < 20000:
        return None

    # Step 1: orient genome by longest-ORF analysis
    genome, longest_orf_start = orient_genome(genome)

    # Step 2: find ORF1a start codon
    orf1a_start0 = find_orf1a_start(genome, longest_orf_start) - 1  # 0-based

    # Step 3: find frameshift site
    slip_pos = find_frameshift_site(genome, orf1a_start0)
    if slip_pos < 0:
        slip_pos = 13462  # last-resort fallback (SARS-CoV-2)

    # Step 4: translate ORF1a (frame 0, to first stop)
    orf1a_nt = Seq(genome[orf1a_start0: slip_pos + len(SLIPPERY_SEQ)])
    pp1a = str(orf1a_nt.translate(to_stop=True))

    # Step 5: translate ORF1b (-1 frameshift, to first stop)
    orf1b_start0 = slip_pos + len(SLIPPERY_SEQ) - 1
    orf1b_end0   = min(slip_pos + 15000, len(genome))
    orf1b_nt = Seq(genome[orf1b_start0: orf1b_end0])
    pp1b = str(orf1b_nt.translate(to_stop=True))

    pp1ab = pp1a + pp1b
    return pp1ab if len(pp1ab) > 3000 else None


def extract_domains_by_coords(pp1ab: str) -> dict[str, str]:
    """
    Extract domain sequences using coordinate-based search.
    Tries SARS-CoV-2 coordinates first; if pp1ab is much shorter/longer,
    scales coordinates proportionally.
    """
    SARS2_LEN = 7098
    scale = len(pp1ab) / SARS2_LEN

    domains = {}
    for dom, (start, end) in DOMAIN_COORDS_SARS2.items():
        s = max(0, int((start - 1) * scale))
        e = min(len(pp1ab), int(end * scale))
        if e > s + 50:
            domains[dom] = pp1ab[s: e]
    return domains


def extract_domains_by_hmm(pp1ab: str, query_id: str = "query") -> dict[str, str]:
    """Extract domain sequences using hmmsearch (more robust for divergent viruses)."""
    if not HMM_FILE.exists():
        return extract_domains_by_coords(pp1ab)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        query_faa = tmp / "query.faa"
        query_faa.write_text(f">{query_id}\n{textwrap.fill(pp1ab, 60)}\n")

        tbl_out = tmp / "hmm_hits.tbl"
        dom_out = tmp / "hmm_dom.tbl"

        cmd = [
            str(HMMER_BIN / "hmmsearch"),
            "--noali", "--domtblout", str(dom_out),
            "--tblout", str(tbl_out),
            "-E", "1e-5",
            str(HMM_FILE), str(query_faa)
        ]
        result = _run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return extract_domains_by_coords(pp1ab)

        # Parse domain table
        domains = {}
        for line in dom_out.read_text().splitlines():
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 20:
                continue
            hmm_name = parts[3]    # HMM name (CoV_3CLpro etc.)
            ali_start = int(parts[17]) - 1  # 0-based
            ali_end   = int(parts[18])
            dom_name = hmm_name.replace("CoV_", "")
            if dom_name in DOMAIN_ORDER:
                hit_len = ali_end - ali_start
                expected = DOMAIN_EXPECTED_SIZE.get(dom_name, 300)
                max_allowed = int(expected * DOMAIN_SIZE_TOLERANCE)
                # Reject hits that are way too large (bad HMM profile issue)
                if hit_len > max_allowed:
                    continue
                score = float(parts[13])
                # Take best hit (highest score) per domain
                if dom_name not in domains or score > domains[dom_name][0]:
                    domains[dom_name] = (score, pp1ab[ali_start: ali_end])

        return {k: v[1] for k, v in domains.items() if len(v[1]) > 50}


def compute_pud(seq1: str, seq2: str) -> float:
    """Compute pairwise uncorrected distance (PUD) between two aligned sequences.
    PUD = fraction of positions with different residues (gaps excluded from denominator)."""
    diff = 0
    total = 0
    for a, b in zip(seq1, seq2):
        if a == "-" and b == "-":
            continue
        if a == "-" or b == "-":
            continue  # ignore gap-vs-residue positions
        total += 1
        if a != b:
            diff += 1
    if total == 0:
        return 1.0
    return diff / total


def align_and_compute_pud(query_domains: dict[str, str],
                           ref_domains: dict[str, str]) -> Optional[float]:
    """Align concatenated domain sequences and compute PUD."""
    # Build concatenated sequences for domains present in both
    common = [d for d in DOMAIN_ORDER if d in query_domains and d in ref_domains]
    if not common:
        return None

    query_concat = "".join(query_domains[d] for d in common)
    ref_concat   = "".join(ref_domains[d]   for d in common)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        pair_fa = tmp / "pair.faa"
        pair_fa.write_text(
            f">query\n{textwrap.fill(query_concat, 60)}\n"
            f">ref\n{textwrap.fill(ref_concat, 60)}\n"
        )
        cmd = [MAFFT_BIN, "--auto", "--quiet", str(pair_fa)]
        result = _run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return None

        # Parse alignment
        seqs = {}
        cur_id = None
        for line in result.stdout.splitlines():
            if line.startswith(">"):
                cur_id = line[1:].split()[0]
                seqs[cur_id] = ""
            elif cur_id:
                seqs[cur_id] += line.strip()

        if "query" not in seqs or "ref" not in seqs:
            return None

        return compute_pud(seqs["query"], seqs["ref"])


def classify_by_pud(pud: float) -> str:
    """Map a PUD value to a taxonomic rank label."""
    if pud <= THRESHOLDS["species"]:
        return "same_species"
    elif pud <= THRESHOLDS["subgenus"]:
        return "same_subgenus"
    elif pud <= THRESHOLDS["genus"]:
        return "same_genus"
    elif pud <= THRESHOLDS["subfamily"]:
        return "same_subfamily"
    elif pud <= THRESHOLDS["family"]:
        return "same_family"
    else:
        return "outside_family"


def get_ref_taxonomy(accession: str) -> dict:
    """Look up taxonomy for a reference accession."""
    import sqlite3
    if not TAX_DB.exists():
        return {}
    acc = accession.split(".")[0]
    try:
        with sqlite3.connect(str(TAX_DB)) as db:
            db.row_factory = sqlite3.Row
            row = db.execute(
                "SELECT * FROM species WHERE species LIKE ? OR species LIKE ? LIMIT 1",
                (f"%{acc}%", f"%{accession}%")
            ).fetchone()
            if row:
                return dict(row)
    except Exception:
        pass
    return {}


def corona_classify_pud(genome_nt: str, top_n: int = 5) -> dict:
    """
    Full pipeline: nucleotide genome → PUD classification.

    Returns:
      {
        "pp1ab_length": int,
        "domains_found": list[str],
        "top_hits": [{"accession": str, "description": str,
                      "pud": float, "rank": str, "taxonomy": dict}],
        "best_classification": {subgenus/genus/subfamily/species},
        "method": "DEmARC_PUD",
        "thresholds": {...}
      }
    """
    # Step 1: translate
    pp1ab = translate_orf1ab(genome_nt)
    if not pp1ab:
        return {"error": "Failed to translate ORF1ab (genome < 20kb or invalid sequence)"}

    # Step 2: extract query domains
    # Use coordinate scaling as primary method (reliable for all pp1ab lengths).
    # HMM can refine individual domains where it gives reasonable-sized hits.
    query_domains = extract_domains_by_coords(pp1ab)
    if not query_domains:
        return {"error": "No domains extracted from ORF1ab"}
    hmm_domains = extract_domains_by_hmm(pp1ab, "query")
    for dom in DOMAIN_ORDER:
        if dom in hmm_domains and dom in query_domains:
            # Only use HMM result if it's a better-sized match
            hmm_len = len(hmm_domains[dom])
            expected = DOMAIN_EXPECTED_SIZE.get(dom, 300)
            if abs(hmm_len - expected) < abs(len(query_domains[dom]) - expected):
                query_domains[dom] = hmm_domains[dom]

    # Step 3: process reference sequences
    if not REF_FASTA.exists():
        return {"error": f"Reference FASTA not found: {REF_FASTA}"}

    hits = []
    for ref_rec in SeqIO.parse(str(REF_FASTA), "fasta"):
        if len(ref_rec.seq) < 20000:
            continue
        ref_pp1ab = translate_orf1ab(str(ref_rec.seq))
        if not ref_pp1ab:
            continue
        ref_domains = extract_domains_by_coords(ref_pp1ab)
        pud = align_and_compute_pud(query_domains, ref_domains)
        if pud is None:
            continue
        hits.append({
            "accession": ref_rec.id,
            "description": ref_rec.description,
            "pud": round(pud, 4),
            "pud_pct": round(pud * 100, 2),
            "rank": classify_by_pud(pud),
        })

    if not hits:
        return {"error": "No PUD computed against any reference"}

    hits.sort(key=lambda x: x["pud"])
    top = hits[:top_n]

    # Step 4: determine best classification from closest hit
    best = hits[0]
    rank = best["rank"]

    # Look up taxonomy for top hits
    import sqlite3
    if TAX_DB.exists():
        with sqlite3.connect(str(TAX_DB)) as db:
            db.row_factory = sqlite3.Row
            for h in top:
                acc = h["accession"].split(".")[0]
                row = db.execute(
                    "SELECT family, subfamily, genus, subgenus, species FROM species "
                    "WHERE species LIKE ? LIMIT 1",
                    (f"%{acc}%",)
                ).fetchone()
                if row:
                    h["taxonomy"] = dict(row)
                else:
                    h["taxonomy"] = {}

    best_tax = top[0].get("taxonomy", {})

    return {
        "pp1ab_length": len(pp1ab),
        "domains_found": list(query_domains.keys()),
        "domain_lengths": {k: len(v) for k, v in query_domains.items()},
        "top_hits": top,
        "best_hit": {
            "accession": best["accession"],
            "pud": best["pud"],
            "pud_pct": best["pud_pct"],
            "rank": rank,
        },
        "classification": {
            "family": "Coronaviridae",
            "subfamily": best_tax.get("subfamily") if rank in ("same_species", "same_subgenus", "same_genus", "same_subfamily") else "unknown",
            "genus":    best_tax.get("genus")    if rank in ("same_species", "same_subgenus", "same_genus") else "unknown",
            "subgenus": best_tax.get("subgenus") if rank in ("same_species", "same_subgenus") else "unknown",
            "species":  best_tax.get("species")  if rank == "same_species" else "novel/uncertain",
        },
        "method": "DEmARC_PUD_5domains",
        "thresholds": THRESHOLDS,
        "note": (
            "PUD thresholds from Ziebuhr et al. 2021 Table 4. "
            "Domains: 3CLpro+NiRAN+RdRp+ZBD+HEL1 (concatenated aa)."
        )
    }
