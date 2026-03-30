#!/usr/bin/env python3
"""
Extract structured ICTV demarcation criteria from family text documents.

Usage:
    python scripts/extract_criteria.py \
        --txt-dir ../ictv_txt \
        --output data/criteria.json \
        [--model claude-sonnet-4-6] \
        [--families coronaviridae flaviviridae ...]  # subset for testing

Output: data/criteria.json — one entry per family.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import anthropic

SYSTEM_PROMPT = """You are an expert virologist specialising in ICTV virus taxonomy.
Your task is to extract structured demarcation criteria from ICTV report chapter text.

Return a JSON object (no markdown fences, raw JSON only) with this exact schema:
{
  "family": "<Family name, CapitalisedCamelCase>",
  "subfamily_demarcation": {
    "primary_method": "<e.g. phylogeny|aa_identity|nt_identity|genetic_distance|genome_organisation|null>",
    "regions": ["<gene or genomic region names>"],
    "thresholds": {"<metric>": <numeric value or null>},
    "description": "<verbatim or close paraphrase of the criteria text, ≤400 chars>"
  },
  "genus_demarcation": {
    "primary_method": "<…>",
    "regions": ["<…>"],
    "thresholds": {"<metric>": <numeric value or null>},
    "description": "<…≤400 chars>"
  },
  "species_demarcation": {
    "primary_method": "<…>",
    "regions": ["<…>"],
    "thresholds": {"<metric>": <numeric value or null>},
    "description": "<…≤400 chars>"
  },
  "reference_gene": "<primary conserved gene used for classification, e.g. RdRp|L|NS5|null>",
  "notes": "<any important caveats or extra info, ≤200 chars or null>"
}

Rules:
- If a level (subfamily/genus/species) is not discussed, set all its fields to null.
- thresholds keys should use: aa_identity_min, nt_identity_min, aa_distance_max,
  nt_distance_max, genetic_distance_max, lwr_min — whichever applies.
  Values are fractions (0-1) for identities/distances, or substitutions/site for distances.
- If no numeric threshold is stated, set thresholds to {}.
- Return ONLY the JSON, no prose."""

USER_TEMPLATE = """Family document: {family_name}

--- BEGIN DOCUMENT ---
{text}
--- END DOCUMENT ---

Extract the demarcation criteria as described."""


def extract_one(client: anthropic.Anthropic, family_name: str, text: str, model: str) -> dict:
    # Truncate very long documents to avoid huge token usage; keep first 12000 chars
    text_trunc = text[:12000]
    msg = client.messages.create(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[
            {"role": "user", "content": USER_TEMPLATE.format(
                family_name=family_name, text=text_trunc)}
        ],
    )
    raw = msg.content[0].text.strip()
    # Strip accidental markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Extract ICTV demarcation criteria via LLM.")
    ap.add_argument("--txt-dir", default="../ictv_txt", help="Directory with *.txt family files")
    ap.add_argument("--output", default="data/criteria.json", help="Output JSON path")
    ap.add_argument("--model", default="claude-sonnet-4-6", help="Claude model ID")
    ap.add_argument("--families", nargs="*", help="Restrict to these family names (no extension)")
    ap.add_argument("--sleep", type=float, default=1.0, help="Sleep between API calls (seconds)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip families already in output file")
    args = ap.parse_args(argv)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        return 1

    client = anthropic.Anthropic(api_key=api_key)
    txt_dir = Path(args.txt_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing results if skip-existing
    existing: dict[str, dict] = {}
    if args.skip_existing and out_path.exists():
        existing = json.loads(out_path.read_text())
        print(f"Loaded {len(existing)} existing entries.")

    # Discover files
    txt_files = sorted(txt_dir.glob("*.txt"))
    if args.families:
        wanted = {f.lower() for f in args.families}
        txt_files = [p for p in txt_files if p.stem.lower() in wanted]

    results: dict[str, dict] = dict(existing)
    ok = 0
    fail = 0

    for txt_path in txt_files:
        family_key = txt_path.stem.lower()  # e.g. "coronaviridae"
        if args.skip_existing and family_key in results:
            print(f"[SKIP] {family_key}")
            continue

        text = txt_path.read_text(encoding="utf-8", errors="replace")
        family_display = family_key.capitalize()
        print(f"[...] {family_display} ", end="", flush=True)
        try:
            data = extract_one(client, family_display, text, args.model)
            data["source_file"] = str(txt_path)
            results[family_key] = data
            print("OK")
            ok += 1
        except Exception as e:
            print(f"FAIL: {e}", file=sys.stderr)
            # Save a stub so we can see what went wrong
            results[family_key] = {"family": family_display, "error": str(e),
                                   "source_file": str(txt_path)}
            fail += 1

        # Save incrementally after each family
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        if args.sleep > 0:
            time.sleep(args.sleep)

    print(f"\nDone: {ok} OK, {fail} failed. Output: {out_path}")
    return 0 if fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
