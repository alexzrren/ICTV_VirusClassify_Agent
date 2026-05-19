#!/usr/bin/env python3
"""
Score the ICTV Agent benchmark run against ground truth.

For each family's Excel output from a benchmark run, compare the agent's
species / genus / family classification against the ground truth TSV
provided in the test dataset.

Usage:
    python scripts/score_benchmark.py benchmark/ /path/to/test_dataset/
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import openpyxl

PROJECT = Path(__file__).resolve().parent.parent


def load_truth(tsv_path: Path) -> dict[str, str]:
    """Parse _testset.sampled.tsv → {accession_version: ground_truth_species}."""
    out: dict[str, str] = {}
    for line in tsv_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2:
            acc = parts[0]          # e.g. MG599871.1
            species = parts[1]      # e.g. Antennavirus hirsutum
            out[acc] = species
    return out


def load_taxonomy_db(db_path: Path):
    """Return a lookup {species_name: {family, genus}} from the MSL species table."""
    lookup: dict[str, dict[str, str]] = {}
    if not db_path.exists():
        return lookup
    db = sqlite3.connect(str(db_path))
    db.row_factory = sqlite3.Row
    for r in db.execute("SELECT species, family, genus FROM species"):
        lookup[r["species"]] = {"family": r["family"] or "", "genus": r["genus"] or ""}
    db.close()
    return lookup


def score_one(
    pred_species: str,
    truth_species: str,
    truth_tax: dict,
    pred_genus: str,
    pred_family: str,
):
    """
    Score a single prediction.
    Returns dict of per-level correctness flags + detail string.
    """
    scores = {
        "species": False,    # exact MSL40 match
        "species_novel": False,  # marked novel AND ground truth says novel (not in MSL)
        "genus": False,
        "family": False,
    }

    ps = (pred_species or "").strip()
    tg = (pred_genus or "").strip()
    pf = (pred_family or "").strip()

    # Species
    is_novel_pred = "sp." in ps.lower() or "novel" in ps.lower()
    truth_tax_exists = bool(truth_tax and truth_tax.get("genus"))

    if is_novel_pred:
        # Agent said novel. Is the ground truth NOT in the MSL species table?
        if not truth_tax_exists:
            scores["species_novel"] = True  # correctly identified as novel
        # For scoring, also check if the genus in the novel placeholder matches truth
        tg_truth = truth_tax.get("genus", "") if truth_tax else ""
        # Extract genus from "Genus sp. (novel, ...)"
        if tg_truth and tg_truth in ps:
            scores["genus"] = True
    else:
        # Exact species match
        if ps.lower() == truth_species.lower():
            scores["species"] = True
        elif truth_tax and truth_tax.get("species", "").lower() == ps.lower():
            scores["species"] = True

    # Genus
    if tg and truth_tax and truth_tax.get("genus", "").lower() == tg.lower():
        scores["genus"] = True

    # Family
    if pf and truth_tax and truth_tax.get("family", "").lower() == pf.lower():
        scores["family"] = True

    return scores


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("benchmark_dir", help="Path to benchmark/ output")
    p.add_argument("--test-dir", default="/home/renzirui/Database/NCBI_Viruses/test_dataset",
                   help="Path to test dataset with *_testset.sampled.tsv files")
    p.add_argument("--db", default=str(PROJECT / "data" / "taxonomy.db"),
                   help="Path to taxonomy.db")
    args = p.parse_args()

    bench = Path(args.benchmark_dir)
    test_dir = Path(args.test_dir)
    db_path = Path(args.db)

    families = sorted(
        d.name for d in bench.iterdir()
        if d.is_dir() and not d.name.startswith("_")
    )
    if not families:
        print("ERROR: No family subdirs found in", bench, file=sys.stderr)
        sys.exit(1)

    print(f"Scoring {len(families)} families: {', '.join(families)}")
    print(f"Taxonomy DB: {db_path}")
    print()

    # Load full taxonomy lookup once
    tax_lookup = load_taxonomy_db(db_path)
    print(f"Taxonomy species loaded: {len(tax_lookup)}")

    # Also load VMR accession lookup for truth genus/family by accession
    vmr_lookup: dict[str, dict] = {}
    if db_path.exists():
        db = sqlite3.connect(str(db_path))
        db.row_factory = sqlite3.Row
        for r in db.execute(
            "SELECT accession, family, subfamily, genus, subgenus, species "
            "FROM vmr_accessions"
        ):
            vmr_lookup[r["accession"]] = dict(r)
        db.close()
    print(f"VMR accession entries: {len(vmr_lookup)}")
    print()

    # ── Per-family scoring ──────────────────────────────────────────────
    total_seqs = 0
    total_species_ok = 0
    total_species_novel_ok = 0
    total_genus_ok = 0
    total_family_ok = 0
    total_errors = 0
    total_medium_low = 0

    family_results: list[dict] = []
    mismatches: list[tuple] = []

    for fam in families:
        xlsx = bench / fam / "results_summary.xlsx"
        tsv = test_dir / f"{fam}_testset.sampled.tsv"
        if not xlsx.exists():
            print(f"  {fam}: SKIP (no xlsx)")
            continue
        if not tsv.exists():
            print(f"  {fam}: WARN — no ground truth TSV")
            continue

        truth = load_truth(tsv)
        wb = openpyxl.load_workbook(str(xlsx), data_only=True)
        ws = wb.active
        headers = [c.value for c in ws[1]]
        col_map = {h.lower(): i for i, h in enumerate(headers)}

        n = 0
        ok_s = 0; ok_sn = 0; ok_g = 0; ok_f = 0
        errors = 0
        med_low = 0

        for row in ws.iter_rows(min_row=2, values_only=True):
            if not row or len(row) <= col_map.get("accession", 0):
                continue
            acc = str(row[col_map["accession"]]).strip()
            status = str(row[col_map.get("status", 1)] or "").strip()
            pred_family = str(row[col_map.get("family", 4)] or "").strip()
            pred_genus = str(row[col_map.get("genus", 5)] or "").strip()
            pred_species = str(row[col_map.get("species", 7)] or "").strip()
            confidence = str(row[col_map.get("confidence", 8)] or "").strip()
            novel = str(row[col_map.get("novel", 9)] or "").strip()

            if status == "error":
                errors += 1
                continue

            if confidence.lower() in ("medium", "low"):
                med_low += 1

            truth_species = truth.get(acc, "")
            # Try bare accession
            if not truth_species:
                bare = acc.split(".")[0]
                truth_species = truth.get(bare, "")

            if not truth_species:
                continue  # can't score without truth

            n += 1
            truth_tax = tax_lookup.get(truth_species, {})
            # Fall back: try VMR lookup on the accession
            if not truth_tax:
                bare = acc.split(".")[0]
                vmr = vmr_lookup.get(bare, {})
                if vmr:
                    truth_tax = {"family": vmr.get("family", ""),
                                 "genus": vmr.get("genus", "")}

            scores = score_one(pred_species, truth_species, truth_tax,
                               pred_genus, pred_family)

            if scores["species"]: ok_s += 1
            if scores["species_novel"]: ok_sn += 1
            if scores["genus"]: ok_g += 1
            if scores["family"]: ok_f += 1

            # Collect mismatches for reporting
            if not any(scores.values()):
                mismatches.append((
                    fam, acc, truth_species,
                    pred_species, pred_genus, pred_family,
                    truth_tax.get("genus", "?"), confidence,
                ))

        total_seqs += n
        total_species_ok += ok_s
        total_species_novel_ok += ok_sn
        total_genus_ok += ok_g
        total_family_ok += ok_f
        total_errors += errors
        total_medium_low += med_low

        species_rate = (ok_s / max(n, 1)) * 100
        species_combined = ((ok_s + ok_sn) / max(n, 1)) * 100
        genus_rate = (ok_g / max(n, 1)) * 100 if n else 0
        family_rate = (ok_f / max(n, 1)) * 100 if n else 0

        family_results.append({
            "family": fam, "n": n, "errors": errors,
            "species": species_rate, "species_combined": species_combined,
            "genus": genus_rate, "family": family_rate,
            "med_low": med_low,
        })

        print(f"{fam:<18s}  n={n:3d}  "
              f"sp={species_rate:5.1f}%  novel+sp={species_combined:5.1f}%  "
              f"genus={genus_rate:5.1f}%  fam={family_rate:5.1f}%  "
              f"err={errors}  medlow={med_low}")

    # ── Global summary ──────────────────────────────────────────────────
    print()
    print("=" * 75)
    print(f"{'GLOBAL':18s}  n={total_seqs:3d}  "
          f"sp={(total_species_ok/total_seqs*100):5.1f}%  "
          f"novel+sp={((total_species_ok+total_species_novel_ok)/total_seqs*100):5.1f}%  "
          f"genus={(total_genus_ok/total_seqs*100):5.1f}%  "
          f"fam={(total_family_ok/total_seqs*100):5.1f}%  "
          f"err={total_errors}")
    print("=" * 75)

    # ── Mismatches ──────────────────────────────────────────────────────
    if mismatches:
        print()
        print(f"MISMATCHES ({len(mismatches)}):")
        for fam, acc, truth_sp, pred_sp, pred_gen, pred_fam, truth_gen, conf in mismatches[:30]:
            print(f"  {fam} {acc}")
            print(f"    truth: {truth_sp} (genus={truth_gen})")
            print(f"    agent: {pred_sp} (genus={pred_gen}, fam={pred_fam}, conf={conf})")

    # ── Token summary ───────────────────────────────────────────────────
    print()
    print("TOKEN USAGE:")
    total_tok_in = 0
    total_tok_out = 0
    total_calls = 0
    total_cached = 0
    for fam in families:
        xlsx = bench / fam / "results_summary.xlsx"
        if not xlsx.exists():
            continue
        wb = openpyxl.load_workbook(str(xlsx), data_only=True)
        ws = wb.active
        headers_list = [(c.value or "").lower() for c in ws[1]]
        idx_in = headers_list.index("tokin") if "tokin" in headers_list else -1
        idx_out = headers_list.index("tokout") if "tokout" in headers_list else -1
        idx_calls = headers_list.index("calls") if "calls" in headers_list else -1
        idx_cached = headers_list.index("cached") if "cached" in headers_list else -1
        fam_in = 0; fam_out = 0; fam_calls = 0; fam_cached = 0
        for row in ws.iter_rows(min_row=2, values_only=True):
            if idx_in >= 0 and row[idx_in]:
                fam_in += int(row[idx_in])
            if idx_out >= 0 and row[idx_out]:
                fam_out += int(row[idx_out])
            if idx_calls >= 0 and row[idx_calls]:
                fam_calls += int(row[idx_calls])
            if idx_cached >= 0 and row[idx_cached] and str(row[idx_cached]).strip().lower() == "yes":
                fam_cached += 1
        total_tok_in += fam_in
        total_tok_out += fam_out
        total_calls += fam_calls
        total_cached += fam_cached
        print(f"  {fam:<18s}  in={fam_in:>7d}  out={fam_out:>7d}  "
              f"calls={fam_calls:>4d}  cached={fam_cached}")
    print(f"  {'TOTAL':18s}  in={total_tok_in:>7d}  out={total_tok_out:>7d}  "
          f"calls={total_calls:>4d}  cached={total_cached}")
    print(f"  Total billed: {total_tok_in + total_tok_out} tokens")


if __name__ == "__main__":
    main()
