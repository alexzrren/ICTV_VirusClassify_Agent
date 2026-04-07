"""
ICTV Classification LLM Agent — ReAct loop using Claude tool_use API.

The agent receives a FASTA sequence and uses a set of bioinformatics tools
together with the ICTV criteria knowledge base to produce a classification.
"""

from __future__ import annotations

import json
import os
import traceback
from pathlib import Path
from typing import AsyncGenerator, Optional

import httpx
import anthropic

from .knowledge.criteria import get_criteria, get_demarcation_summary, list_families
from .knowledge.rag import semantic_search
from .models import ClassifyResult, Evidence, TaxonomyResult
from .tools.alignment import parse_fasta, pairwise_identity
from .tools.blast import blastn, diamond_blastp
from .tools.taxonomy import full_taxonomy, lookup_by_family, lookup_species, search_any_level
from .tools.corona_pud import corona_classify_pud

# ── Tool definitions (Claude tool_use schema) ───────────────────────────────

TOOLS = [
    {
        "name": "blast_search",
        "description": (
            "Search a query nucleotide or protein sequence against ICTV reference sequences "
            "using BLAST/DIAMOND. Returns top hits with family, accession, % identity, "
            "e-value, and query coverage. Use this as the first step to identify the "
            "likely virus family."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sequence": {"type": "string", "description": "Raw sequence (no FASTA header)"},
                "seq_type": {
                    "type": "string",
                    "enum": ["nucleotide", "protein"],
                    "description": "Sequence type",
                },
                "max_hits": {"type": "integer", "default": 10},
            },
            "required": ["sequence", "seq_type"],
        },
    },
    {
        "name": "get_criteria",
        "description": (
            "Retrieve the ICTV demarcation criteria for a virus family. "
            "Returns structured criteria including the primary method, genomic regions, "
            "numerical thresholds, and a description from the ICTV report chapter."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "family": {"type": "string", "description": "Virus family name, e.g. Coronaviridae"},
                "level": {
                    "type": "string",
                    "enum": ["species", "genus", "subfamily", "all"],
                    "default": "all",
                    "description": "Which demarcation level to retrieve",
                },
            },
            "required": ["family"],
        },
    },
    {
        "name": "compute_pairwise_identity",
        "description": (
            "Compute pairwise sequence identity (%) between two sequences using MAFFT alignment. "
            "Use this to apply identity-threshold demarcation criteria."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "seq1": {"type": "string", "description": "First raw sequence"},
                "seq2": {"type": "string", "description": "Second raw sequence"},
                "is_protein": {
                    "type": "boolean",
                    "default": False,
                    "description": "True if sequences are amino acid",
                },
            },
            "required": ["seq1", "seq2"],
        },
    },
    {
        "name": "lookup_taxonomy",
        "description": (
            "Look up the full ICTV taxonomy (Realm → Species) for a virus name or accession "
            "from the MSL40 database. Supports fuzzy matching."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Virus species name or accession"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_ictv_docs",
        "description": (
            "Search ICTV report chapter documents for relevant text about a topic. "
            "Useful for looking up biological properties, genome organisation, or additional "
            "classification context not captured in the structured criteria."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language search query"},
                "n_results": {"type": "integer", "default": 3},
            },
            "required": ["query"],
        },
    },
    {
        "name": "list_reference_species",
        "description": (
            "List species in a given virus family or genus from the ICTV MSL40. "
            "Useful for selecting reference sequences for identity calculations."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "family": {"type": "string", "description": "Family name (optional)"},
                "genus": {"type": "string", "description": "Genus name (optional)"},
            },
        },
    },
    {
        "name": "fetch_reference_sequence",
        "description": (
            "Fetch a reference sequence from the local ICTV reference database by accession "
            "or keyword. Returns the raw sequence for use in pairwise identity calculations. "
            "Use after blast_search to retrieve the full reference sequence of a BLAST hit."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "accession": {
                    "type": "string",
                    "description": "GenBank accession (e.g. NC_045512) or partial header keyword",
                },
                "family": {
                    "type": "string",
                    "description": "Family name to narrow search (optional)",
                },
            },
            "required": ["accession"],
        },
    },
    {
        "name": "blast_and_compare",
        "description": (
            "All-in-one tool: BLAST the query sequence against references, then compute "
            "pairwise identity with the top N hits. Returns BLAST hits with their pairwise "
            "identity to the query. This is the most efficient way to classify a sequence."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "sequence": {"type": "string", "description": "Raw query sequence"},
                "seq_type": {
                    "type": "string",
                    "enum": ["nucleotide", "protein"],
                    "description": "Sequence type",
                },
                "top_n": {
                    "type": "integer",
                    "default": 5,
                    "description": "Number of top hits to compute pairwise identity for",
                },
            },
            "required": ["sequence", "seq_type"],
        },
    },
    {
        "name": "corona_pud_classify",
        "description": (
            "Coronaviridae-specific subgenus/genus/subfamily/species classification using the "
            "DEmARC PUD (Pairwise Uncorrected Distance) method. Translates ORF1ab from the "
            "nucleotide genome, extracts 5 conserved replicase domains (3CLpro, NiRAN, RdRp, "
            "ZBD, HEL1), and computes PUD against all Coronaviridae reference sequences. "
            "Applies ICTV Table 4 thresholds (Ziebuhr et al. 2021) to assign rank. "
            "Use this tool when the query sequence is identified as Coronaviridae and you need "
            "subgenus-level resolution."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "genome_nt": {
                    "type": "string",
                    "description": "Complete (or near-complete) coronavirus nucleotide genome sequence, >20 kb",
                },
                "top_n": {
                    "type": "integer",
                    "default": 5,
                    "description": "Number of top reference hits to return",
                },
            },
            "required": ["genome_nt"],
        },
    },
]


# ── Tool execution ───────────────────────────────────────────────────────────

def _execute_tool(name: str, inputs: dict) -> str:
    """Dispatch tool call and return result as JSON string."""
    try:
        if name == "blast_search":
            seq = inputs["sequence"].replace("\n", "").replace(" ", "")
            seq_type = inputs.get("seq_type", "nucleotide")
            max_hits = int(inputs.get("max_hits", 10))
            if seq_type == "protein":
                hits = diamond_blastp(seq, max_hits=max_hits)
            else:
                hits = blastn(seq, max_hits=max_hits)
            result = [
                {
                    "rank": i + 1,
                    "subject_id": h.subject_id,
                    "pident": h.pident,
                    "evalue": h.evalue,
                    "bitscore": h.bitscore,
                    "qcovs": h.qcovs,
                    "description": h.stitle,
                }
                for i, h in enumerate(hits)
            ]
            if not result:
                return json.dumps({"hits": [], "message": "No significant BLAST hits found."})
            return json.dumps({"hits": result})

        elif name == "get_criteria":
            family = inputs["family"]
            level = inputs.get("level", "all")
            crit = get_criteria(family)
            if not crit:
                # Try to find the closest family name
                families = list_families()
                fam_lower = family.lower()
                close = [f for f in families if fam_lower in f or f in fam_lower]
                if close:
                    crit = get_criteria(close[0])
                    if crit:
                        crit["_note"] = f"Matched to '{close[0]}' (requested: '{family}')"
            if not crit:
                return json.dumps({
                    "error": f"No criteria found for '{family}'.",
                    "available_families": list_families()[:20],
                })
            if level == "all":
                return json.dumps(crit)
            key = f"{level}_demarcation"
            return json.dumps({
                "family": crit.get("family", family),
                "level": level,
                key: crit.get(key),
                "reference_gene": crit.get("reference_gene"),
                "notes": crit.get("notes"),
            })

        elif name == "compute_pairwise_identity":
            s1 = inputs["seq1"].replace("\n", "").replace(" ", "")
            s2 = inputs["seq2"].replace("\n", "").replace(" ", "")
            is_prot = bool(inputs.get("is_protein", False))
            if len(s1) < 10 or len(s2) < 10:
                return json.dumps({"error": "Sequences too short for identity calculation."})
            pident = pairwise_identity(s1, s2, is_protein=is_prot)
            return json.dumps({
                "pairwise_identity": round(pident * 100, 2),
                "fraction": round(pident, 4),
                "seq1_len": len(s1),
                "seq2_len": len(s2),
            })

        elif name == "lookup_taxonomy":
            query = inputs["query"]
            rows = lookup_species(query)
            if not rows:
                rows = search_any_level(query)
            if not rows:
                return json.dumps({"message": f"No taxonomy found for '{query}'."})
            # Return top 5, omitting None values
            clean = [{k: v for k, v in r.items() if v is not None} for r in rows[:5]]
            return json.dumps({"results": clean})

        elif name == "search_ictv_docs":
            query = inputs["query"]
            n = int(inputs.get("n_results", 3))
            hits = semantic_search(query, n_results=n)
            return json.dumps([
                {"family": h["family"], "score": round(h["score"], 3),
                 "text": h["text"][:600]}
                for h in hits
            ])

        elif name == "list_reference_species":
            family = inputs.get("family", "")
            genus = inputs.get("genus", "")
            if family:
                rows = lookup_by_family(family)
            elif genus:
                from .tools.taxonomy import lookup_by_genus
                rows = lookup_by_genus(genus)
            else:
                return json.dumps({"error": "Provide family or genus."})
            clean = [
                {"species": r["species"], "genus": r["genus"], "family": r["family"]}
                for r in rows[:30]
            ]
            return json.dumps({"count": len(rows), "species": clean})

        elif name == "fetch_reference_sequence":
            acc = inputs["accession"].strip()
            family = inputs.get("family", "")
            ref_dir = Path(__file__).resolve().parent.parent / "data" / "references"
            # Search in specific family or all families
            search_dirs = []
            if family:
                fam_dir = ref_dir / family
                if fam_dir.exists():
                    search_dirs.append(fam_dir)
            if not search_dirs:
                search_dirs = [d for d in ref_dir.iterdir() if d.is_dir()]
            # Search FASTA files for matching accession
            for fam_dir in search_dirs:
                fasta_file = fam_dir / "sequences.fasta"
                if not fasta_file.exists():
                    continue
                seqs = parse_fasta(fasta_file.read_text())
                for header, seq in seqs.items():
                    if acc.lower() in header.lower():
                        return json.dumps({
                            "accession": header.split()[0],
                            "header": header[:200],
                            "family": fam_dir.name,
                            "length": len(seq.replace("\n", "")),
                            "sequence": seq.replace("\n", ""),
                        })
            return json.dumps({"error": f"No reference sequence found for '{acc}'."})

        elif name == "blast_and_compare":
            seq = inputs["sequence"].replace("\n", "").replace(" ", "")
            seq_type = inputs.get("seq_type", "nucleotide")
            top_n = int(inputs.get("top_n", 5))
            # Step 1: BLAST
            if seq_type == "protein":
                hits = diamond_blastp(seq, max_hits=top_n)
            else:
                hits = blastn(seq, max_hits=top_n)
            if not hits:
                return json.dumps({"hits": [], "message": "No BLAST hits found."})
            # Step 2: For each hit, fetch reference seq and compute pairwise identity
            ref_dir = Path(__file__).resolve().parent.parent / "data" / "references"
            results = []
            all_ref_seqs = {}  # cache
            for fam_dir in ref_dir.iterdir():
                if not fam_dir.is_dir():
                    continue
                fasta_file = fam_dir / "sequences.fasta"
                if not fasta_file.exists():
                    continue
                for header, rseq in parse_fasta(fasta_file.read_text()).items():
                    all_ref_seqs[header] = (fam_dir.name, rseq.replace("\n", ""))
            # Deduplicate: keep best HSP per subject
            seen_subjects = {}
            for h in hits:
                sid = h.subject_id
                if sid not in seen_subjects or h.bitscore > seen_subjects[sid].bitscore:
                    seen_subjects[sid] = h
            unique_hits = list(seen_subjects.values())[:top_n]

            for h in unique_hits:
                acc = h.subject_id.split("|")[-1]  # strip family prefix
                hit_info = {
                    "subject_id": h.subject_id,
                    "accession": acc,
                    "blast_pident": h.pident,
                    "evalue": h.evalue,
                    "bitscore": h.bitscore,
                    "description": h.stitle,
                }
                # Find matching ref sequence
                ref_seq = None
                for header, (fam, rseq) in all_ref_seqs.items():
                    if acc in header:
                        hit_info["family"] = fam
                        ref_seq = rseq
                        break
                if ref_seq and len(seq) >= 10 and len(ref_seq) >= 10:
                    is_prot = seq_type == "protein"
                    try:
                        pid = pairwise_identity(seq, ref_seq, is_protein=is_prot)
                        hit_info["global_pairwise_identity"] = round(pid * 100, 2)
                    except Exception as e:
                        hit_info["identity_error"] = str(e)
                results.append(hit_info)
            return json.dumps({"hits": results, "count": len(results)})

        elif name == "corona_pud_classify":
            genome_nt = inputs["genome_nt"].replace("\n", "").replace(" ", "")
            top_n = int(inputs.get("top_n", 5))
            result = corona_classify_pud(genome_nt, top_n=top_n)
            return json.dumps(result)

        else:
            return json.dumps({"error": f"Unknown tool: {name}"})

    except FileNotFoundError as e:
        return json.dumps({"error": f"Tool unavailable: {e}. "
                           "Reference databases may not be built yet."})
    except Exception as e:
        return json.dumps({"error": str(e), "traceback": traceback.format_exc()[-500:]})


# ── System prompt ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert ICTV virus taxonomist and bioinformatician.
Your task is to classify a virus sequence by strictly following ICTV official demarcation criteria.

## Workflow (follow these steps IN ORDER)

**Step 1: Identify family and compute identity**
- Use blast_and_compare — this is the MOST IMPORTANT tool. It BLASTs the query against all references AND computes global pairwise identity with top hits in one call.
- If family_hint is provided, still run blast_and_compare to get quantitative identity values.

**Step 2: Get ICTV criteria**
- Use get_criteria to retrieve the official ICTV demarcation criteria for the identified family.

**Step 3: Compare and classify**
- Compare the global_pairwise_identity values from blast_and_compare against the ICTV thresholds.
- Use lookup_taxonomy to get the full taxonomy of the best matching reference species.
- Determine if the query is: same species, same genus but new species, or more divergent.
- For Coronaviridae: use corona_pud_classify (instead of or in addition to blast_and_compare) to get subgenus-level resolution via DEmARC PUD on the 5 replicase domains.

**Step 4: Output result**
- Output the final JSON classification.

## Key rules
- ALWAYS call blast_and_compare first — it provides both family identification AND pairwise identity in one step.
- Use the global_pairwise_identity (not blast_pident) for ICTV threshold comparison, as BLAST pident is local alignment only.
- For Coronaviridae sequences >20 kb, call corona_pud_classify to get subgenus classification using DEmARC PUD thresholds.
- Cite specific ICTV criteria thresholds and how your computed values compare.
- Confidence levels: High (identity clearly above/below threshold), Medium (near threshold ±5%), Low (insufficient data).

## Available families in criteria knowledge base (32 families):
coronaviridae, picornaviridae, paramyxoviridae, flaviviridae, togaviridae, caliciviridae,
rhabdoviridae, adenoviridae, hantaviridae, filoviridae, orthoherpesviridae, arenaviridae,
polyomaviridae, papillomaviridae, amnoonviridae, anelloviridae, arteriviridae, asfarviridae,
astroviridae, bornaviridae, circoviridae, hepeviridae, nairoviridae, nodaviridae, parvoviridae,
peribunyaviridae, phenuiviridae, picobirnaviridae, pneumoviridae, poxviridae, sedoreoviridae,
spinareoviridae

## Final output format
At the end, ALWAYS output a JSON block (inside ```json ... ```) with this EXACT structure:
```json
{
  "query_id": "...",
  "taxonomy": {
    "realm": "...", "kingdom": "...", "phylum": "...", "class": "...",
    "order": "...", "family": "...", "subfamily": "...", "genus": "...", "species": "..."
  },
  "confidence": "High|Medium|Low",
  "novel_species": true/false,
  "evidence": [
    {"method": "blast_and_compare", "region": "whole genome", "value": 92.5, "threshold": 90, "conclusion": "Above species threshold"}
  ],
  "reasoning": "Brief explanation of classification logic"
}
```"""


# ── Agent loop ───────────────────────────────────────────────────────────────

def _check_blast_available() -> bool:
    """Check if BLAST databases exist."""
    from .tools.blast import BLASTN_DB, DIAMOND_DB
    from pathlib import Path
    blastn_ok = any(Path(f"{BLASTN_DB}{ext}").exists() for ext in [".nhr", ".nin", ".nsq"])
    diamond_ok = Path(DIAMOND_DB).exists()
    return blastn_ok or diamond_ok


async def classify_sequence(
    fasta: str,
    max_steps: int = 20,
    step_callback=None,
    family_hint: str = "",
) -> tuple[ClassifyResult, list[str]]:
    """
    Run the classification agent.

    step_callback: optional async callable(step_text: str) called after each agent step.
    family_hint: optional family name to skip BLAST and go directly to criteria lookup.
    Returns (ClassifyResult, list_of_step_logs).
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")

    base_url = os.environ.get("ANTHROPIC_BASE_URL")

    # Temporarily clear proxy env vars so httpx doesn't use SOCKS proxy
    _proxy_vars = {}
    for pv in ("http_proxy", "https_proxy", "all_proxy",
               "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        if pv in os.environ:
            _proxy_vars[pv] = os.environ.pop(pv)

    client_kwargs = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = anthropic.Anthropic(**client_kwargs)

    # Restore proxy env vars for other tools (e.g. NCBI Entrez)
    os.environ.update(_proxy_vars)

    # Parse FASTA to get query IDs and sequences
    seqs = parse_fasta(fasta)
    if not seqs:
        raise ValueError("No valid FASTA sequences found in input.")

    # For now process first sequence
    query_id, sequence = next(iter(seqs.items()))
    raw_seq = sequence.replace("\n", "").replace(" ", "")

    step_logs: list[str] = []

    def log(msg: str):
        step_logs.append(msg)

    # Check tool availability
    blast_available = _check_blast_available()
    avail_note = ""
    if not blast_available:
        avail_note = (
            "\n\n**NOTE: BLAST reference databases are NOT yet built.** "
            "blast_search will return an error. Focus on using get_criteria, "
            "lookup_taxonomy, and search_ictv_docs instead. "
            "Explain what computations would be needed for proper classification."
        )

    family_note = ""
    if family_hint:
        family_note = (
            f"\n\n**The user has indicated this sequence belongs to family: {family_hint}.** "
            f"Skip blast_search and go directly to get_criteria for {family_hint}."
        )

    user_message = (
        f"Please classify the following virus sequence.\n\n"
        f"Query ID: {query_id}\n"
        f"Sequence length: {len(raw_seq)} bp/aa\n\n"
        f"Sequence (first 500 chars): {raw_seq[:500]}{'...' if len(raw_seq) > 500 else ''}\n\n"
        f"Full sequence available in tools as needed. "
        f"Use the tools to identify the virus family and classify it according to ICTV criteria."
        f"{family_note}{avail_note}"
    )

    messages = [{"role": "user", "content": user_message}]

    # Store full sequence for tool use (injected into subsequent tool calls)
    _full_seq = raw_seq

    final_result: Optional[ClassifyResult] = None
    tool_result_store: list[tuple[str, str]] = []  # (tool_name, full_result_json)

    import asyncio

    for step in range(max_steps):
        # Run synchronous API call in thread pool to avoid blocking event loop
        response = await asyncio.to_thread(
            client.messages.create,
            model=os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6"),
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        # Collect text content for logging
        text_parts = []
        tool_uses = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_uses.append(block)

        if text_parts:
            combined_text = "\n".join(text_parts)
            log(f"[Step {step+1}] {combined_text[:300]}")
            if step_callback:
                await step_callback(f"[Step {step+1}] {combined_text[:300]}")

            # Try to extract final JSON result
            if "```json" in combined_text and '"taxonomy"' in combined_text:
                try:
                    json_str = combined_text.split("```json")[1].split("```")[0].strip()
                    data = json.loads(json_str)
                    final_result = _parse_result(data, query_id)
                except Exception:
                    pass

        # Append assistant message
        messages.append({"role": "assistant", "content": response.content})

        log(f"[Debug] stop_reason={response.stop_reason}, tool_uses={len(tool_uses)}")

        if response.stop_reason == "end_turn":
            break

        if response.stop_reason == "tool_use":
            # Execute tools
            tool_results = []
            for tu in tool_uses:
                log(f"[Tool] {tu.name}({json.dumps({k: str(v)[:100] for k, v in tu.input.items()})})")
                if step_callback:
                    await step_callback(f"[Tool] Calling {tu.name}...")

                # Always inject full sequence for BLAST tools
                inputs = dict(tu.input)
                if tu.name in ("blast_search", "blast_and_compare"):
                    inputs["sequence"] = _full_seq

                result_str = await asyncio.to_thread(_execute_tool, tu.name, inputs)
                tool_result_store.append((tu.name, result_str))
                log(f"[Tool result] {result_str[:200]}")

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": result_str,
                })

            messages.append({"role": "user", "content": tool_results})

    # If no structured result extracted, try to build one from tool results
    if final_result is None:
        final_result = _build_result_from_logs(query_id, step_logs, tool_result_store)

    return final_result, step_logs


def _build_result_from_logs(
    query_id: str,
    step_logs: list[str],
    tool_results: list[tuple[str, str]],
) -> ClassifyResult:
    """Extract classification info from tool results when model didn't output JSON."""
    import re

    taxonomy = TaxonomyResult()
    evidence = []
    confidence = "Low"
    reasoning_parts = []

    # Parse all tool results for structured data
    all_tool_data = " ".join(r for _, r in tool_results)

    # Extract taxonomy from lookup_taxonomy results
    tax_match = re.search(
        r'"realm":\s*"([^"]*)".*?"kingdom":\s*"([^"]*)".*?"phylum":\s*"([^"]*)".*?'
        r'"class":\s*"([^"]*)".*?"order":\s*"([^"]*)".*?"family":\s*"([^"]*)".*?'
        r'"genus":\s*"([^"]*)".*?"species":\s*"([^"]*)"',
        all_tool_data,
    )
    if tax_match:
        taxonomy = TaxonomyResult(
            realm=tax_match.group(1) or None,
            kingdom=tax_match.group(2) or None,
            phylum=tax_match.group(3) or None,
            **{"class": tax_match.group(4) or None},
            order=tax_match.group(5) or None,
            family=tax_match.group(6) or None,
            genus=tax_match.group(7) or None,
            species=tax_match.group(8) or None,
        )
        confidence = "Medium"

    # Extract BLAST + pairwise identity from blast_and_compare results
    for tool_name, result_str in tool_results:
        if tool_name in ("blast_and_compare", "blast_search"):
            try:
                data = json.loads(result_str)
                hits = data.get("hits", [])
                for h in hits[:3]:
                    if "global_pairwise_identity" in h:
                        evidence.append(Evidence(
                            method="global_pairwise_identity",
                            region="whole genome",
                            value=h["global_pairwise_identity"],
                            threshold=None,
                            conclusion=f"vs {h.get('subject_id','?')}: {h['global_pairwise_identity']}%",
                        ))
                        confidence = "Medium"
                    if "blast_pident" in h:
                        evidence.append(Evidence(
                            method="blastn",
                            region="local alignment",
                            value=h["blast_pident"],
                            threshold=None,
                            conclusion=f"vs {h.get('subject_id','?')}: {h['blast_pident']}%",
                        ))
            except (json.JSONDecodeError, KeyError):
                pass

    # Extract step text for reasoning
    for log_line in step_logs:
        if log_line.startswith("[Step"):
            reasoning_parts.append(log_line)

    return ClassifyResult(
        query_id=query_id,
        taxonomy=taxonomy,
        confidence=confidence,
        novel_species=False,
        evidence=evidence,
        reasoning=" | ".join(reasoning_parts) if reasoning_parts else "See agent steps for details.",
    )


def _parse_result(data: dict, query_id: str) -> ClassifyResult:
    tax = data.get("taxonomy", {})
    taxonomy = TaxonomyResult(
        realm=tax.get("realm"),
        kingdom=tax.get("kingdom"),
        phylum=tax.get("phylum"),
        **{"class": tax.get("class")},
        order=tax.get("order"),
        family=tax.get("family"),
        subfamily=tax.get("subfamily"),
        genus=tax.get("genus"),
        species=tax.get("species"),
    )
    evidence = [
        Evidence(
            method=e.get("method", ""),
            region=e.get("region", ""),
            value=e.get("value"),
            threshold=e.get("threshold"),
            conclusion=e.get("conclusion", ""),
        )
        for e in data.get("evidence", [])
    ]
    return ClassifyResult(
        query_id=data.get("query_id", query_id),
        taxonomy=taxonomy,
        confidence=data.get("confidence", "Unknown"),
        novel_species=bool(data.get("novel_species", False)),
        evidence=evidence,
        reasoning=data.get("reasoning", ""),
    )
