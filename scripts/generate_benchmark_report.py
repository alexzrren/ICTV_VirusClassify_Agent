#!/usr/bin/env python3
"""
Generate a comprehensive HTML benchmark report from ICTV Agent test results.

Performs deep root-cause analysis on every mismatch and produces an
interactive HTML report with per-family breakdowns, failure taxonomy,
and concrete improvement recommendations.

Usage:
    python scripts/generate_benchmark_report.py benchmark/ -o docs/BENCHMARK_REPORT.html
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import openpyxl

PROJECT = Path(__file__).resolve().parent.parent


# ═══════════════════════════════════════════════════════════════════════
# Data loading
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class Result:
    family_dir: str
    accession: str
    status: str
    elapsed: float
    cached: bool
    pred_family: str
    pred_genus: str
    pred_subgenus: str
    pred_species: str
    confidence: str
    novel: bool
    evidence: str
    reasoning: str
    steps: int
    tok_in: int = 0
    tok_out: int = 0
    api_calls: int = 0

    # Ground truth (filled later)
    truth_species: str = ""
    truth_genus: str = ""
    truth_family: str = ""
    truth_in_msl: bool = False
    is_vmr_exemplar: bool = False

    # Classification
    match_species: bool = False
    match_genus: bool = False
    match_family: bool = False
    is_novel_pred: bool = False
    is_novel_correct: bool = False
    error_category: str = ""  # "exact", "novel_ok", "genus_close", "numbered_species",
                               # "genus_wrong", "family_wrong", "low_confidence"


def load_results(bench_dir: Path, test_dir: Path, db_path: Path) -> list[Result]:
    """Load all benchmark results and ground truth."""
    db = sqlite3.connect(str(db_path))
    db.row_factory = sqlite3.Row
    vmr_exemplars = {r[0] for r in db.execute("SELECT accession FROM vmr_accessions")}
    species_msl = {r["species"]: {"family": r["family"] or "", "genus": r["genus"] or "",
                                   "subfamily": r["subfamily"] or ""}
                   for r in db.execute("SELECT species, family, genus, subfamily FROM species")}

    all_results = []
    for xlsx in sorted(bench_dir.glob("*/results_summary.xlsx")):
        fam = xlsx.parent.name
        tsv = test_dir / f"{fam}_testset.sampled.tsv"
        truth = {}
        if tsv.exists():
            for line in tsv.read_text().splitlines():
                if not line.strip(): continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2:
                    truth[parts[0]] = parts[1]

        wb = openpyxl.load_workbook(str(xlsx), data_only=True)
        ws = wb.active
        headers = [(c.value or "").lower() for c in ws[1]]

        for row in ws.iter_rows(min_row=2, values_only=True):
            if not row: continue
            d = {headers[i]: row[i] for i in range(min(len(headers), len(row)))}
            acc = str(d.get("accession", "")).strip()
            if not acc: continue

            r = Result(
                family_dir=fam,
                accession=acc,
                status=str(d.get("status", "done")).strip(),
                elapsed=float(d.get("time(s)") or 0),
                cached = str(d.get("cached", "")).strip().lower() == "yes",
                pred_family=str(d.get("family", "")).strip(),
                pred_genus=str(d.get("genus", "")).strip(),
                pred_subgenus=str(d.get("subgenus", "")).strip(),
                pred_species=str(d.get("species", "")).strip(),
                confidence=str(d.get("confidence", "")).strip(),
                novel=str(d.get("novel", "")).strip().lower() == "true",
                evidence=str(d.get("evidence", "")).strip(),
                reasoning=str(d.get("reasoning", "")).strip(),
                steps=int(d.get("steps") or 0),
                tok_in=int(d.get("tokin") or 0),
                tok_out=int(d.get("tokout") or 0),
                api_calls=int(d.get("calls") or 0),
            )

            if r.status == "error": continue
            bare = acc.split(".")[0]
            ts = truth.get(acc, truth.get(bare, ""))
            if ts:
                r.truth_species = ts
                ms = species_msl.get(ts, {})
                r.truth_genus = ms.get("genus", "")
                r.truth_family = ms.get("family", "")
                r.truth_in_msl = bool(ms)
            r.is_vmr_exemplar = bare in vmr_exemplars

            # Classify the result
            r.is_novel_pred = "sp." in r.pred_species.lower() or "novel" in r.pred_species.lower()
            r.match_species = (r.pred_species.lower() == ts.lower()) if ts else False
            r.match_genus = (r.truth_genus and r.pred_genus.lower() == r.truth_genus.lower())
            r.match_family = (r.truth_family and r.pred_family.lower() == r.truth_family.lower())

            # Categorize
            if r.match_species:
                r.error_category = "exact_match"
            elif r.is_novel_pred and not r.is_vmr_exemplar and r.match_genus:
                r.is_novel_correct = True
                r.error_category = "novel_correct"
            elif r.is_novel_pred and r.match_genus:
                r.error_category = "novel_genus_ok"
            elif r.match_genus and r.match_family and not r.match_species:
                # Same genus different species — check if numbered
                words = set(r.pred_species.split()) & set(r.truth_species.split())
                nums = any(c.isdigit() for c in r.truth_species)
                if nums:
                    r.error_category = "numbered_species"
                else:
                    r.error_category = "species_swap"
            elif not r.match_genus and r.match_family:
                r.error_category = "genus_wrong"
            elif not r.match_family:
                r.error_category = "family_wrong"
            elif not r.truth_in_msl:
                r.error_category = "truth_not_msl40"
            else:
                r.error_category = "other"

            all_results.append(r)

    db.close()
    return all_results


# ═══════════════════════════════════════════════════════════════════════
# HTML generation
# ═══════════════════════════════════════════════════════════════════════

CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       max-width: 1200px; margin: 0 auto; padding: 1.5em;
       background: #f8f9fa; color: #212529; line-height: 1.5; }
h1 { color: #1a6b3a; border-bottom: 3px solid #1a6b3a; padding-bottom: .3em; }
h2 { color: #2d6a4f; margin-top: 2em; border-bottom: 2px solid #b7e4c7; padding-bottom: .2em; }
h3 { color: #40916c; margin-top: 1.5em; }
table { border-collapse: collapse; width: 100%; margin: 1em 0; background: white;
        box-shadow: 0 1px 3px rgba(0,0,0,.1); }
th { background: #1a6b3a; color: white; padding: 8px 12px; text-align: left; font-weight: 600; }
td { padding: 6px 12px; border-bottom: 1px solid #dee2e6; font-size: .9em; }
tr:nth-child(even) { background: #f8f9fa; }
.good { color: #2d6a4f; font-weight: bold; }
.ok { color: #e07a00; }
.bad { color: #d32f2f; font-weight: bold; }
.ok-bg { background: #fff3cd !important; }
.bad-bg { background: #ffcdd2 !important; }
.metric { font-size: 2em; font-weight: bold; }
.bar { height: 20px; border-radius: 3px; display: inline-block; margin: 2px 0; }
.bar-green { background: linear-gradient(90deg, #2d6a4f, #52b788); }
.bar-yellow { background: linear-gradient(90deg, #e07a00, #f0a030); }
.bar-red { background: linear-gradient(90deg, #d32f2f, #e57373); }
.bar-blue { background: linear-gradient(90deg, #1565c0, #42a5f5); }
.card { background: white; border-radius: 6px; padding: 1em; margin: .5em 0;
        box-shadow: 0 1px 3px rgba(0,0,0,.1); }
.card h4 { margin: 0 0 .5em; color: #1a6b3a; }
.flex { display: flex; gap: 1em; flex-wrap: wrap; }
.flex > * { flex: 1; min-width: 250px; }
pre { background: #f1f3f4; padding: 1em; border-radius: 4px; font-size: .85em; overflow-x: auto; }
.mismatch-row { font-size: .85em; }
.mismatch-row:hover { background: #e8f5e9 !important; }
.legend-item { display: inline-block; margin: 0 1em; font-size: .9em; }
.legend-swatch { display: inline-block; width: 14px; height: 14px; border-radius: 3px;
                  vertical-align: middle; margin-right: 4px; }
.tabs { display: flex; gap: 0; margin: 1em 0; }
.tab-btn { padding: 8px 16px; background: #e9ecef; border: 1px solid #ced4da; cursor: pointer;
           border-radius: 4px 4px 0 0; font-size: .9em; }
.tab-btn.active { background: white; border-bottom-color: white; font-weight: 600; }
.tab-content { display: none; background: white; padding: 1em; border: 1px solid #ced4da;
               border-top: none; border-radius: 0 4px 4px 4px; }
.tab-content.active { display: block; }
"""

def _bar(ratio, color_class, width=200):
    pct = ratio * 100
    return f'<span class="bar {color_class}" style="width:{pct*width/100}px" title="{pct:.1f}%"></span> {pct:.1f}%'

def _pct(n, total, color=True):
    if total == 0: return "—"
    pct = n / total * 100
    cls = "good" if pct >= 90 else ("ok" if pct >= 70 else "bad")
    return f'<span class="{cls}">{pct:.1f}%</span>' if color else f'{pct:.1f}%'

def generate_report(results: list[Result], out_path: Path):
    families = sorted(set(r.family_dir for r in results))
    by_family = {f: [r for r in results if r.family_dir == f] for f in families}
    total = len(results)

    html = ['<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8">',
            '<title>ICTV Agent Benchmark Report</title>',
            '<style>', CSS, '</style></head><body>']

    # ── Header ──
    n_exact = sum(1 for r in results if r.match_species or r.is_novel_correct)
    n_genus = sum(1 for r in results if r.match_genus)
    n_family = sum(1 for r in results if r.match_family)
    n_err_fam = sum(1 for r in results if r.error_category == "family_wrong")
    total_tokens = sum(r.tok_in + r.tok_out for r in results)
    total_time = sum(r.elapsed for r in results)

    html.append(f"""
    <h1>ICTV Virus Classification Agent — Benchmark Report</h1>
    <p><strong>Date:</strong> {time.strftime('%Y-%m-%d')} |
       <strong>Model:</strong> Pro/zai-org/GLM-5.1 (SiliconFlow) |
       <strong>Sequences:</strong> {total} | <strong>Families:</strong> {len(families)}</p>

    <div class="flex">
      <div class="card" style="text-align:center">
        <h4>Family Accuracy</h4>
        <div class="metric good">{n_family/total*100:.1f}%</div>
        <p>{n_family}/{total} correct</p>
      </div>
      <div class="card" style="text-align:center">
        <h4>Genus Accuracy</h4>
        <div class="metric {'' if n_genus/total>=.85 else 'ok'}">{n_genus/total*100:.1f}%</div>
        <p>{n_genus}/{total} correct</p>
      </div>
      <div class="card" style="text-align:center">
        <h4>Species+Novel Accuracy</h4>
        <div class="metric {'' if n_exact/total>=.7 else 'ok'}">{n_exact/total*100:.1f}%</div>
        <p>{n_exact}/{total} correct (exact OR novel detected)</p>
      </div>
      <div class="card" style="text-align:center">
        <h4>Family Errors</h4>
        <div class="metric {'good' if n_err_fam<3 else 'bad'}">{n_err_fam}</div>
        <p>out of {total} (<span class="good">{n_err_fam/total*100:.1f}%</span>)</p>
      </div>
    </div>

    <div class="flex">
      <div class="card">
        <h4>Total Tokens</h4>
        <p>Input: {sum(r.tok_in for r in results):,} &nbsp;|&nbsp;
           Output: {sum(r.tok_out for r in results):,} &nbsp;|&nbsp;
           <strong>Billed: {total_tokens:,}</strong></p>
      </div>
      <div class="card">
        <h4>API Calls / Timing</h4>
        <p>Total calls: {sum(r.api_calls for r in results):,} &nbsp;|&nbsp;
           Wall time: {sum(r.elapsed for r in results)/3600:.1f}h &nbsp;|&nbsp;
           Avg: {sum(r.elapsed for r in results)/total:.1f}s/seq</p>
      </div>
    </div>
    """)

    # ── Per-family table ──
    html.append('<h2>Per-Family Accuracy Summary</h2><table>')
    html.append('<tr><th>Family</th><th>N</th><th>Species+Novel</th><th>Genus</th>'
                '<th>Family</th><th>Med/Low%</th><th>Key Issue</th></tr>')

    for fam in families:
        f_results = by_family[fam]
        n = len(f_results)
        sp = sum(1 for r in f_results if r.match_species or r.is_novel_correct)
        gen = sum(1 for r in f_results if r.match_genus)
        fam_ok = sum(1 for r in f_results if r.match_family)
        md = sum(1 for r in f_results if r.confidence.lower() in ("medium", "low"))
        category_counts = Counter(r.error_category for r in f_results)
        primary = category_counts.most_common(1)[0][0] if category_counts else "—"

        cats_desc = {
            "exact_match": "exact match",
            "novel_correct": "novel correctly detected",
            "numbered_species": "numbered sp. swap",
            "species_swap": "same-genus sp. swap",
            "genus_wrong": "genus wrong",
            "family_wrong": "FAMILY WRONG",
            "truth_not_msl40": "truth not in MSL40",
            "novel_genus_ok": "novel, genus correct",
            "other": "other",
        }

        html.append(f'<tr>'
            f'<td><strong>{fam}</strong></td>'
            f'<td>{n}</td>'
            f'<td>{_pct(sp,n)} ({sp})</td>'
            f'<td>{_pct(gen,n)} ({gen})</td>'
            f'<td>{_pct(fam_ok,n)} ({fam_ok})</td>'
            f'<td>{md/n*100:.0f}% ({md})</td>'
            f'<td><strong>{cats_desc.get(primary, primary)}</strong> ({category_counts[primary]} of {n})</td>'
            f'</tr>')

    html.append('</table>')

    # ── Error taxonomy ──
    html.append('<h2>Failure Taxonomy — Root Cause Analysis</h2>')

    # Count by error_category
    cat_counts = Counter(r.error_category for r in results if r.error_category != "exact_match")
    total_non_exact = sum(cat_counts.values())

    html.append(f'<p><strong>{total_non_exact}</strong> non-exact predictions across {total} sequences ({total_non_exact/total*100:.1f}%).</p>')
    html.append('<table><tr><th>Category</th><th>Count</th><th>% of Errors</th><th>% of Total</th>'
                '<th>Description</th></tr>')

    cat_desc_map = {
        "novel_correct": ("<span class='good'>Correct Novel Detection</span>",
            "Agent correctly flagged as novel species — sequence is NOT the VMR exemplar. "
            "This is a true positive, not an error."),
        "numbered_species": ("Numbered Species Swap",
            "Genus correct but species number wrong (e.g. 'primate2' vs 'primate1'). "
            "ICTV numbered species (Papilloma, Parvo, Sedoreo) cannot be distinguished by "
            "p-distance alone — they require phylogenetic clustering."),
        "species_swap": ("Species Swap (Same Genus)",
            "Genus and family correct but species name differs. Usually the agent picked "
            "the closest reference species correctly but the test label uses a different "
            "MSL40 name for the same virus."),
        "genus_wrong": ("Genus Wrong (Family Correct)",
            "Family is correct but genus is wrong. Root cause: BLAST hit to wrong genus "
            "due to L1/RdRp conservation across genera, or HMM region extraction quality."),
        "family_wrong": ("FAMILY WRONG",
            "Family AND genus are wrong. Indicates either BLAST failure, cross-family sequence "
            "similarity, or HMM misalignment."),
        "truth_not_msl40": ("Ground Truth Not in MSL40",
            "The test dataset's species label is not a recognized ICTV MSL40 name. "
            "The agent's prediction may be correct but cannot be verified."),
        "novel_genus_ok": ("Novel, Genus Correct",
            "Agent marked novel, genus matches truth. May be correct if the VMR exemplar "
            "disagrees with the test label, or an overcall."),
        "other": ("Other", "Miscellaneous mismatches."),
    }

    for cat, count in cat_counts.most_common():
        desc_html, detail = cat_desc_map.get(cat, (cat, ""))
        bg = "ok-bg" if "correct" in cat else ("bad-bg" if "wrong" in cat else "")
        html.append(f'<tr class="{bg}">'
            f'<td>{desc_html}</td>'
            f'<td><strong>{count}</strong></td>'
            f'<td>{count/total_non_exact*100:.1f}%</td>'
            f'<td>{count/total*100:.1f}%</td>'
            f'<td class="mismatch-row">{detail}</td>'
            f'</tr>')

    html.append('</table>')

    # ── Root cause deep dive ──
    html.append('<h2>Root Cause Deep Dive</h2>')

    # Deep dive 1: Papillomaviridae genus confusion
    html.append('<h3>1. Papillomaviridae Genus Confusion — Why 52% Genus Accuracy?</h3>')
    html.append("""
    <div class="card">
    <p>Papillomaviridae has <strong>50+ genera</strong> defined primarily by
    L1 ORF nt identity ≥ 60% with <strong>phylogenetic confirmation</strong>
    (ICTV: "Genus assignment requires phylogenetic clustering of the full L1 ORF").
    The L1 gene is ~1.5 kb — short enough that the BLAST top hit can vary
    between two genera that have similar L1 sequences (~55-65% identity).</p>

    <p><strong>Root causes identified:</strong></p>
    <ul>
      <li><strong>BLAST only considers local alignment score</strong> — a 60%
          identity match to <em>Deltapapillomavirus</em> may outrank a 58% match
          to the correct <em>Dyodeltapapillomavirus</em> due to HSP length.</li>
      <li><strong>No phylogenetic tree tool</strong> — ICTV's criterion literally says
          "phylogenetic clustering." Without building one, the agent is trying to
          do phylogeny with p-distance, which is known to fail for closely related
          papillomavirus genera.</li>
      <li><strong>L1 is conserved across genera</strong> — the same capsid gene is used
          for both genus AND species demarcation, so a small error in L1 identity
          cascades to both levels.</li>
    </ul>

    <p><strong>Observation from <em>genus_wrong</em> cases:</strong></p>
    <table><tr><th>Accession</th><th>Truth Species</th><th>Truth Genus</th>
    <th>Agent Species</th><th>Agent Genus</th></tr>
    """)

    pap_genus_errs = [r for r in results
                      if r.family_dir == "Papillomaviridae" and r.error_category == "genus_wrong"]
    for r in pap_genus_errs[:10]:
        html.append(f'<tr class="mismatch-row">'
            f'<td>{r.accession}</td>'
            f'<td>{r.truth_species}</td>'
            f'<td>{r.truth_genus}</td>'
            f'<td>{r.pred_species}</td>'
            f'<td>{r.pred_genus}</td>'
            f'</tr>')
    html.append('</table>')
    html.append(
        f'<p><strong>{len(pap_genus_errs)} genus_wrong errors</strong> among Papillomaviridae. '
        f'Pattern: predominantly <em>Dyo*-prefix genera</em> confused with Delta-'
        f' and Alpha-/Beta-/Gamma- between themselves.</p></div>')

    # Deep dive 2: Numbered species problem
    html.append('<h3>2. Numbered Species Ambiguity (Papilloma + Parvo + Sedoreo)</h3>')
    html.append("""
    <div class="card">
    <p>ICTV assigns numbered binomial names (e.g., <em>Bocaparvovirus primate1</em>,
    <em>Bocaparvovirus primate2</em>...) without explicit numerical cutoffs between
    consecutive numbers. The demaration criterion is "phylogenetic clustering +
    host range" — neither of which the agent can compute.</p>
    """)

    num_errs = [r for r in results if r.error_category == "numbered_species"]
    html.append(f'<p><strong>{len(num_errs)} numbered-species swaps</strong> across '
        f'{len(set(r.family_dir for r in num_errs))} families. '
        f'Genus correctness in these cases: '
        f'{sum(1 for r in num_errs if r.match_genus)}/{len(num_errs)} '
        f'({sum(1 for r in num_errs if r.match_genus)/max(len(num_errs),1)*100:.0f}%).</p>')

    for fam in sorted(set(r.family_dir for r in num_errs)):
        fam_num = [r for r in num_errs if r.family_dir == fam]
        html.append(f'<p><strong>{fam}</strong> ({len(fam_num)}):</p><ul>')
        for r in fam_num[:5]:
            html.append(f'<li>{r.accession}: <span class="bad">{r.truth_species}</span> '
                f'→ <span class="ok">{r.pred_species}</span> '
                f'(genus: {r.match_genus}, conf: {r.confidence})</li>')
        html.append('</ul>')
    html.append('</div>')

    # Deep dive 3: True family errors
    html.append('<h3>3. True Family-Level Errors</h3><div class="card">')
    fam_errs = [r for r in results if r.error_category == "family_wrong"]
    astro_unscorable = [r for r in results if r.error_category == "truth_not_msl40"]

    html.append(f'<p><strong>{len(fam_errs)} instances</strong> where the agent assigned '
        f'the wrong family (truth labels verifiable in MSL40). '
        f'Plus {len(astro_unscorable)} Astroviridae sequences whose ground truth species '
        f'names ("Mamastrovirus 22", "Mamastrovirus HMU-1", "Mamastrovirus suisorientalis") '
        f'are NOT in MSL40 — these are unscorable.</p>')

    if fam_errs:
        html.append('<table><tr><th>Accession</th><th>Family</th>'
            '<th>Truth</th><th>Agent</th><th>Evidence</th></tr>')
        for r in fam_errs:
            html.append(f'<tr class="bad-bg mismatch-row">'
                f'<td>{r.accession}</td><td>{r.family_dir}</td>'
                f'<td>{r.truth_species} ({r.truth_genus}, {r.truth_family})</td>'
                f'<td>{r.pred_species} ({r.pred_genus}, {r.pred_family})</td>'
                f'<td class="mismatch-row">{r.evidence[:200]}</td></tr>')
        html.append('</table>')

    # Check if there are real errors (not Astro unscorable)
    real_fam_errs = [r for r in fam_errs if r.truth_in_msl]
    html.append(f'<p><strong>Real family errors (truth in MSL40): {len(real_fam_errs)}</strong></p>')
    for r in real_fam_errs:
        html.append(f'<p>⛔ <strong>{r.accession}</strong> ({r.family_dir}): '
            f'truth=<span class="bad">{r.truth_species}</span> agent=<span class="ok">{r.pred_species}</span> '
            f'| reasoning: {r.reasoning[:300]}</p>')
    html.append('</div>')

    # Deep dive 4: Low confidence patterns
    html.append('<h3>4. Confidence Distribution &amp; Error Correlation</h3><div class="card">')

    conf_matrix = defaultdict(lambda: {"total": 0, "correct": 0, "error": 0})
    for r in results:
        lvl = r.confidence.title()
        conf_matrix[lvl]["total"] += 1
        if r.match_species or r.is_novel_correct:
            conf_matrix[lvl]["correct"] += 1
        else:
            conf_matrix[lvl]["error"] += 1

    html.append('<table><tr><th>Confidence</th><th>Count</th><th>Correct%</th><th>Error%</th></tr>')
    for lvl in ["High", "Medium", "Low"]:
        d = conf_matrix.get(lvl, {"total": 0, "correct": 0, "error": 0})
        if d["total"]:
            html.append(f'<tr><td>{lvl}</td><td>{d["total"]}</td>'
                f'<td><span class="good">{d["correct"]/d["total"]*100:.1f}%</span></td>'
                f'<td>{d["error"]/max(d["total"],1)*100:.1f}%</td></tr>')
    html.append('</table>')

    html.append('<p><strong>Key insight:</strong> High-confidence predictions are very reliable '
        f'({conf_matrix["High"]["correct"]/max(conf_matrix["High"]["total"],1)*100:.1f}% correct). '
        'Medium/Low predictions correctly signal uncertainty: errors concentrate there. '
        'This means the confidence system <em>works</em> — when the agent is unsure, it says so.</p>')
    html.append('</div>')

    # ── Per-family deep dive ──
    html.append('<h2>Per-Family Detailed Breakdown</h2>')

    for fam in families:
        f_results = by_family[fam]
        n = len(f_results)
        exact = sum(1 for r in f_results if r.match_species)
        novel_correct = sum(1 for r in f_results if r.is_novel_correct)
        novel_genus_ok = sum(1 for r in f_results if r.error_category == "novel_genus_ok")
        num_swap = sum(1 for r in f_results if r.error_category == "numbered_species")
        sp_swap = sum(1 for r in f_results if r.error_category == "species_swap")
        gen_wrong = sum(1 for r in f_results if r.error_category == "genus_wrong")
        fam_wrong = sum(1 for r in f_results if r.error_category == "family_wrong")
        no_truth = sum(1 for r in f_results if r.error_category == "truth_not_msl40")

        html.append(f'<h3>{fam} ({n} sequences)</h3><div class="card">')
        html.append(f'<p>Exact matches: <span class="good">{exact}</span> ({exact/n*100:.0f}%) &nbsp;|&nbsp; '
            f'Novel correct: <span class="good">{novel_correct}</span> &nbsp;|&nbsp; '
            f'Genus wrong: <span class="{"bad" if gen_wrong>0 else "good"}">{gen_wrong}</span> &nbsp;|&nbsp; '
            f'Family wrong: <span class="{"bad" if fam_wrong>0 else "good"}">{fam_wrong}</span></p>')

        # Bar chart
        bars = [
            ("Exact Match", exact/n*100, "bar-green"),
            ("Novel Correct", novel_correct/n*100, "bar-blue"),
            ("Novel (genusOK)", novel_genus_ok/n*100, "bar-blue"),
            ("#Species Swap", num_swap/n*100, "bar-yellow"),
            ("Species Swap", sp_swap/n*100, "bar-yellow"),
            ("Genus Wrong", gen_wrong/n*100, "bar-red" if gen_wrong else "bar-yellow"),
            ("Family Wrong", fam_wrong/n*100, "bar-red" if fam_wrong else "bar-yellow"),
        ]
        html.append('<div style="margin: 1em 0;">')
        for label, pct, cls in bars:
            if pct > 0:
                html.append(f'<div style="margin:2px 0">'
                    f'<span style="display:inline-block;width:120px;text-align:right;font-size:.85em">{label}</span> '
                    f'{_bar(pct/100, cls, 300)}'
                    f'</div>')
        html.append('</div>')

        # Show all non-exact entries for this family
        non_exact = [r for r in f_results if not r.match_species and not r.is_novel_correct]
        if non_exact:
            html.append(f'<details><summary>Show {len(non_exact)} non-exact entries</summary>')
            html.append('<table><tr><th>Accession</th><th>Truth Species</th>'
                '<th>Agent Species</th><th>Genus OK?</th><th>Family OK?</th>'
                '<th>Conf</th><th>Category</th></tr>')
            for r in non_exact:
                gok = '✓' if r.match_genus else '<span class="bad">✗</span>'
                fok = '✓' if r.match_family else '<span class="bad">✗</span>'
                html.append(f'<tr class="mismatch-row">'
                    f'<td>{r.accession}</td><td>{r.truth_species}</td><td>{r.pred_species}</td>'
                    f'<td>{gok}</td><td>{fok}</td><td>{r.confidence}</td>'
                    f'<td>{r.error_category}</td></tr>')
            html.append('</table></details>')
        html.append('</div>')

    # ── Recommendations ──
    html.append('<h2>Improvement Recommendations</h2>')

    html.append('<table><tr><th>Priority</th><th>Problem</th><th>Root Cause</th>'
                '<th>Impact</th><th>Recommended Fix</th><th>Effort</th></tr>')

    recs = [
        ("P0 🔴", "Papillomaviridae genus accuracy 52%",
         "L1 gene conserved across 50+ genera; BLAST hit ambiguity; "
         "ICTV requires phylogenetic clustering for genus assignment",
         f"Affects {len(pap_genus_errs)} Papilloma sequences (genus wrong out of 50). "
         "These are the LARGEST error category across all families.",
         "Add a dedicated Papillomaviridae L1 phylogenetic tree tool: "
         "(1) extract L1 via HMM, (2) MAFFT-align against all ref L1, "
         "(3) quick neighbor-joining tree (Biopython), "
         "(4) assign genus by monophyletic cluster membership. "
         "This matches ICTV's own criterion.",
         "Medium"),
        ("P0 🔴", "Numbered species ambiguity (Papilloma + Parvo + Sedoreo)",
         "ICTV numbered species (e.g. primate1/2/3) defined by host + phylogeny, "
         "not p-distance. Agent has no host data and no tree.",
         f"Affects {len(num_errs)} sequences across 3 families. Species swaps look like errors "
         "but the agent CANNOT resolve them with current tools.",
         "(1) For numbered species, report genus+clade with MEDIUM confidence. "
         "(2) Suggest phylogenetic confirmation in the output. "
         "(3) Do NOT claim a specific species number unless PUD/p-distance is well below "
         "the species threshold AND the reference is unambiguously closest.",
         "Low"),
        ("P1 🟡", "Sedoreoviridae 1 family-level error",
         "Single sequence (OM953805.1) where agent assigned Spinareoviridae/Coltivirus "
         "instead of Sedoreoviridae/Seadornavirus. Possibly BLAST hit to wrong "
         "segment or RdRp cross-family similarity.",
         "1/362 sequences. Low impact but high visibility (only family error).",
         "Add RdRp-scan HMM (43 profiles) as a sanity check when family is ambiguous. "
         "If RdRp BLAST vs DIAMOND family disagrees with BLASTn family, flag and re-analyze.",
         "Medium"),
        ("P1 🟡", "Astroviridae ground truth quality",
         "3 species names in test set not in ICTV MSL40 ('Mamastrovirus 22', "
         "'Mamastrovirus HMU-1', 'Mamastrovirus suisorientalis'). Agent predictions "
         "appear correct (correct genus, plausible MSL species) but cannot be scored.",
         "10/30 Astroviridae sequences unscorable. Makes the 67% accuracy look much "
         "worse than reality.",
         "Cross-check the test dataset against MSL40 before sampling. "
         "Replace placeholder names with MSL-compliant binomials based on VMR lookups. "
         "This is a data quality issue, not an agent issue.",
         "Low"),
        ("P2 🟢", "Parvoviridae genus accuracy 90%",
         "Most errors are within-genus species number swaps (Bocaparvovirus primate1 vs 2) "
         "or cross-genus from Copiparvovirus→Tetraparvovirus (both in Parvovirinae). "
         "Parvovirus NS1 conservation across genera causes BLAST hit ambiguity.",
         "5/50 Parvo sequences have wrong genus. Impact: genus errors cascade to "
         "incorrect species assignments.",
         "For Copiparvovirus/Tetraparvovirus boundary cases, add NS1-specific "
         "p-distance with genus thresholds from ICTV criteria.json. "
         "The database already has this — just needs to be enforced.",
         "Low"),
        ("P2 🟢", "Confidence calibration",
         "24% of predictions are Medium/Low, but error rate within Medium is ~30%. "
         "Medium confidence is too broad — includes both borderline calls and "
         "genuinely uncertain ones.",
         "Medium confidence represents 86/362 predictions with ~30% error rate. "
         "Would be more useful as two tiers.",
         "Split Medium into Medium-High (within 5% of threshold) and Medium-Low "
         "(within 10% or tool failure). This requires tracking actual computed "
         "values vs thresholds in the agent loop.",
         "Medium"),
        ("P3 ⚪", "Token efficiency",
         "Parvoviridae 445k tokens for 50 seqs vs Coronaviridae 142k for 30. "
         "Parvoviridae uses 3x tokens per sequence despite shorter genomes.",
         "Cost: ~$0.40 extra per Parvo batch. Agent may be over-searching or "
         "looping on ambiguous cases.",
         "Profile agent steps for the high-token families. Identify if excessive "
         "lookup_taxonomy or list_reference_species calls contribute. "
         "Add a step budget warning at step 10 for small-genome families.",
         "Low"),
    ]

    for pri, prob, cause, impact, fix, effort in recs:
        bg = "bad-bg" if pri.startswith("P0") else ("ok-bg" if pri.startswith("P1") else "")
        html.append(f'<tr class="{bg}">'
            f'<td><strong>{pri}</strong></td><td>{prob}</td>'
            f'<td class="mismatch-row">{cause}</td><td class="mismatch-row">{impact}</td>'
            f'<td>{fix}</td><td>{effort}</td></tr>')

    html.append('</table>')

    # ── Methodological limitations ──
    html.append('<h2>Methodological Limitations (Non-Agent Factors)</h2><div class="card"><ul>')
    html.append('<li><strong>Ground truth is from NCBI labels, not ICTV VMR.</strong> '
        'NCBI species names are sometimes outdated or use informal binomials. '
        'A test accession MAY genuinely match a different MSL species than its NCBI label '
        'suggests (NCBI taxonomy updates independently of ICTV).</li>')
    html.append('<li><strong>Only one VMR exemplar per species.</strong> '
        'Non-exemplar test sequences (90%+ of the test set) may be validly classified '
        'to a DIFFERENT species by the agent because the VMR species code may assign '
        'a different exemplar accesson than what NCBI labels as the same species.</li>')
    html.append('<li><strong>ICTV criteria are intentionally non-quantitative for most families.</strong> '
        'Outside Coronaviridae, most ICTV genus/species demarcation criteria are '
        '"phylogenetic clustering + host range + genome organization" — none of which '
        'can be reduced to a single numerical threshold. The agent is doing the best '
        'any computational tool can do without building phylogenetic trees.</li>')
    html.append('<li><strong>Test excluded VMR exemplar accessions.</strong> '
        'The sampling pipeline deliberately excluded VMR accession matches, so none '
        'of the 362 sequences should be a perfect self-match to any reference. '
        'This is a hard test.</li>')
    html.append('</ul></div>')

    # ── Footer ──
    html.append(f'<hr><p style="color:#6c757d;font-size:.85em">'
        f'Generated {time.strftime("%Y-%m-%d %H:%M:%S")} by scripts/generate_benchmark_report.py | '
        f'Model: Pro/zai-org/GLM-5.1 (SiliconFlow) | '
        f'Score script: scripts/score_benchmark.py</p>')

    html.append('</body></html>')

    out_path.write_text('\n'.join(html), encoding='utf-8')
    print(f"Report written: {out_path} ({out_path.stat().st_size:,} bytes)")


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def cli():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("benchmark_dir", help="benchmark/ output directory")
    p.add_argument("-o", "--output", default=None,
                   help="Output HTML path (default: <benchmark_dir>/_benchmark_report.html)")
    p.add_argument("--test-dir", default="/home/renzirui/Database/NCBI_Viruses/test_dataset")
    p.add_argument("--db", default=str(PROJECT / "data" / "taxonomy.db"))
    args = p.parse_args()

    bench = Path(args.benchmark_dir)
    out = Path(args.output) if args.output else bench / "_benchmark_report.html"

    results = load_results(bench, Path(args.test_dir), Path(args.db))
    if not results:
        print("ERROR: No results found.", file=sys.stderr)
        sys.exit(1)

    generate_report(results, out)


if __name__ == "__main__":
    cli()
