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

# ── Shared LLM client (reused across classify calls to avoid TCP cold-start) ──

_shared_client: anthropic.Anthropic | None = None
_shared_client_key: tuple[str, str] = ("", "")  # (api_key, base_url)


def _get_shared_client(api_key: str, base_url: str | None) -> anthropic.Anthropic:
    """Return a module-level Anthropic client, creating or recycling as needed.

    The httpx.Client is configured with trust_env=False (bypasses local proxy)
    and a generous 120-second timeout. Keeping a single client across classify
    calls lets httpx reuse its TCP/TLS connection pool, eliminating the 60-130s
    cold-start penalty on first API call after server restart.
    """
    global _shared_client, _shared_client_key
    key = (api_key, base_url or "")
    if _shared_client is not None and _shared_client_key == key:
        return _shared_client

    http_client = httpx.Client(trust_env=False, timeout=120.0)
    kwargs: dict = {"api_key": api_key, "http_client": http_client}
    if base_url:
        kwargs["base_url"] = base_url
    _shared_client = anthropic.Anthropic(**kwargs)
    _shared_client_key = key
    return _shared_client


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
    {
        "name": "extract_target_region",
        "description": (
            "Extract family-specific target protein region(s) from a nucleotide genome "
            "using HMM profiles. Supports most families with defined classification regions "
            "(e.g. L protein for Paramyxoviridae, VP1 for Caliciviridae, NS5 RdRp for "
            "Flaviviridae, etc.). Returns extracted protein sequences that can then be used "
            "with compute_pairwise_identity for ICTV threshold comparison. "
            "Use this when get_criteria specifies a particular gene region for comparison.\n\n"
            "Two modes:\n"
            "  • Default: extract from the QUERY genome (genome_nt is auto-injected server-side).\n"
            "  • Reference mode: set ref_accession to the accession of a reference you previously "
            "fetched via fetch_reference_sequence. The server will resolve the full cached "
            "reference sequence and extract from it. Use this to get the SAME region from a "
            "reference so you can then call compute_pairwise_identity(query_region, ref_region)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "family": {
                    "type": "string",
                    "description": "Virus family name (e.g. Paramyxoviridae)",
                },
                "ref_accession": {
                    "type": "string",
                    "description": (
                        "Optional. If set, extract from the cached reference sequence with "
                        "this accession instead of the query. The accession must have been "
                        "fetched previously via fetch_reference_sequence."
                    ),
                },
            },
            "required": ["family"],
        },
    },
    {
        "name": "compare_query_to_reference",
        "description": (
            "All-in-one region comparison tool. Given a virus family and a reference accession "
            "(from BLAST hits), this tool:\n"
            "  1. Extracts family-specific target protein regions from the QUERY genome\n"
            "  2. Fetches the reference genome from the local database\n"
            "  3. Extracts the SAME regions from the reference\n"
            "  4. Computes amino acid pairwise identity and p-distance for each region\n\n"
            "Returns aa_identity (%) and aa_p_distance for each region, ready for ICTV "
            "threshold comparison. Use this instead of manually chaining extract_target_region "
            "and compute_pairwise_identity — it's faster and avoids passing large protein "
            "sequences through the conversation.\n\n"
            "Example: For Hepacivirus (Flaviviridae), ICTV requires NS3 p-distance >0.25 and "
            "NS5B p-distance >0.30 for novel species. Call compare_query_to_reference with "
            "family='Flaviviridae' and ref_accession='KX905133.1' to get both p-distances "
            "in one call."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "family": {
                    "type": "string",
                    "description": "Virus family name (e.g. Flaviviridae, Paramyxoviridae)",
                },
                "ref_accession": {
                    "type": "string",
                    "description": "Reference accession from a BLAST hit (e.g. KX905133.1)",
                },
            },
            "required": ["family", "ref_accession"],
        },
    },
]


# ── Tool execution ───────────────────────────────────────────────────────────

# Cache reference sequences fetched during a classification run so they can
# be auto-injected into downstream tools (extract_target_region, pairwise id)
# without flowing back through the LLM context.
_ref_seq_cache: dict[str, str] = {}


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
                        clean_seq = seq.replace("\n", "")
                        # Cache full sequence for auto-injection into downstream tools
                        _ref_seq_cache[header.split()[0]] = clean_seq
                        return json.dumps({
                            "accession": header.split()[0],
                            "header": header[:200],
                            "family": fam_dir.name,
                            "length": len(clean_seq),
                            "sequence_preview": clean_seq[:80] + "..." + clean_seq[-40:],
                            "note": (
                                "Full sequence is cached server-side. blast_and_compare "
                                "already provides global_pairwise_identity for BLAST hits; "
                                "DO NOT attempt to manually re-run compute_pairwise_identity "
                                "on reference vs query with raw sequences — use the values "
                                "already reported by blast_and_compare."
                            ),
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

        elif name == "extract_target_region":
            from .tools.hmmer import extract_all_regions, list_available_hmms
            family = inputs["family"]
            ref_accession = (inputs.get("ref_accession") or "").strip()
            source_label = "query"
            if ref_accession:
                # Reference mode: resolve cached full-length sequence.
                genome_nt = _ref_seq_cache.get(ref_accession, "")
                if not genome_nt:
                    # Fallback: scan reference FASTA files for the accession.
                    ref_dir = Path(__file__).resolve().parent.parent / "data" / "references"
                    for fam_dir in ref_dir.iterdir():
                        if not fam_dir.is_dir():
                            continue
                        fasta_file = fam_dir / "sequences.fasta"
                        if not fasta_file.exists():
                            continue
                        for header, seq in parse_fasta(fasta_file.read_text()).items():
                            if ref_accession.lower() in header.lower():
                                genome_nt = seq.replace("\n", "")
                                _ref_seq_cache[header.split()[0]] = genome_nt
                                break
                        if genome_nt:
                            break
                if not genome_nt:
                    return json.dumps({
                        "error": (
                            f"Reference accession '{ref_accession}' not found in cache "
                            f"or reference database. Call fetch_reference_sequence first."
                        )
                    })
                source_label = f"reference:{ref_accession}"
            else:
                genome_nt = inputs["genome_nt"].replace("\n", "").replace(" ", "")
            regions = extract_all_regions(genome_nt, family)
            if not regions:
                avail = list_available_hmms()
                return json.dumps({
                    "error": f"No target regions extracted for {family}",
                    "available_families": list(avail.keys()),
                })
            return json.dumps({
                "source": source_label,
                "family": family,
                "regions_extracted": list(regions.keys()),
                "region_lengths": {k: len(v) for k, v in regions.items()},
                "sequences": {k: v[:50] + "..." if len(v) > 50 else v
                              for k, v in regions.items()},
                "note": (
                    "Protein sequences truncated for display. "
                    "Use compare_query_to_reference for p-distance calculation."
                ),
            })

        elif name == "compare_query_to_reference":
            from .tools.hmmer import extract_all_regions_with_nt
            family = inputs["family"]
            ref_accession = inputs["ref_accession"].strip()

            # 1. Extract regions from query (genome_nt is auto-injected)
            query_nt = inputs.get("genome_nt", "").replace("\n", "").replace(" ", "")
            query_regions = extract_all_regions_with_nt(query_nt, family)
            if not query_regions:
                return json.dumps({
                    "error": f"No target regions extracted from query for {family}",
                })

            # 2. Resolve reference genome
            ref_nt = _ref_seq_cache.get(ref_accession, "")
            if not ref_nt:
                ref_dir = Path(__file__).resolve().parent.parent / "data" / "references"
                for fam_dir in ref_dir.iterdir():
                    if not fam_dir.is_dir():
                        continue
                    fasta_file = fam_dir / "sequences.fasta"
                    if not fasta_file.exists():
                        continue
                    for header, seq in parse_fasta(fasta_file.read_text()).items():
                        if ref_accession.lower() in header.lower():
                            ref_nt = seq.replace("\n", "")
                            _ref_seq_cache[header.split()[0]] = ref_nt
                            break
                    if ref_nt:
                        break
            if not ref_nt:
                return json.dumps({
                    "error": f"Reference '{ref_accession}' not found in local database.",
                })

            # 3. Extract same regions from reference
            ref_regions = extract_all_regions_with_nt(ref_nt, family)
            if not ref_regions:
                return json.dumps({
                    "error": f"No target regions extracted from reference {ref_accession}",
                })

            # 4. Compute pairwise identity for each matching region (both aa AND nt)
            comparisons = {}
            for region_name, q_reg in query_regions.items():
                if region_name not in ref_regions:
                    comparisons[region_name] = {
                        "error": "Region not found in reference",
                        "query_aa_length": len(q_reg.protein),
                    }
                    continue
                r_reg = ref_regions[region_name]
                entry: dict = {
                    "query_aa_length": len(q_reg.protein),
                    "ref_aa_length": len(r_reg.protein),
                }
                # Amino acid identity
                try:
                    aa_id = pairwise_identity(q_reg.protein, r_reg.protein, is_protein=True)
                    entry["aa_identity_pct"] = round(aa_id * 100, 2)
                    entry["aa_p_distance"] = round(1.0 - aa_id, 4)
                except Exception as e:
                    entry["aa_error"] = str(e)
                # Nucleotide identity (if both nt subsequences are available)
                if q_reg.nucleotide and r_reg.nucleotide:
                    entry["query_nt_length"] = len(q_reg.nucleotide)
                    entry["ref_nt_length"] = len(r_reg.nucleotide)
                    try:
                        nt_id = pairwise_identity(
                            q_reg.nucleotide, r_reg.nucleotide, is_protein=False)
                        entry["nt_identity_pct"] = round(nt_id * 100, 2)
                    except Exception as e:
                        entry["nt_error"] = str(e)
                comparisons[region_name] = entry

            return json.dumps({
                "family": family,
                "ref_accession": ref_accession,
                "query_regions_found": list(query_regions.keys()),
                "ref_regions_found": list(ref_regions.keys()),
                "comparisons": comparisons,
            })

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
- **Family identification MUST be based on BLAST hits only.** NEVER infer family from genome size, GC content, or other heuristics.
- **When ICTV criteria specify an amino-acid or nucleotide threshold on a SPECIFIC gene region (Hepacivirus NS3/NS5B, Paramyxoviridae L protein, Picornaviridae P1, Flaviviridae NS5, Papillomaviridae L1, etc.), you MUST compute that specific distance — do NOT substitute whole-genome BLAST identity.** Use:
    compare_query_to_reference(family=X, ref_accession=Y)
    This single tool call extracts the required protein region(s) from BOTH the query and the reference genome, computes amino acid pairwise identity and p-distance for each region, and returns structured results ready for threshold comparison. No need to chain extract_target_region + compute_pairwise_identity manually.
- **Do not invent thresholds.** If get_criteria returns "no numerical threshold (phylogeny + ecology)" for the genus (e.g. Orthoflavivirus), SAY SO in your output and use BLAST hit + species name from lookup_taxonomy as evidence. Do NOT fabricate arbitrary nt-identity thresholds like "75%/70%".
- **When the ICTV threshold is on whole genome** (e.g. Papillomaviridae L1 nt identity, Polyomaviridae genome identity), use blast_pident or global_pairwise_identity from blast_and_compare directly.
- **Budget your tool calls.** Simple species-level confirmations need ~3 tool calls. Region-based p-distance classifications need ~4 calls (blast_and_compare → get_criteria → compare_query_to_reference → lookup_taxonomy). If you find yourself beyond 8 calls without conclusion, stop and report what you have.
- For Coronaviridae sequences >20 kb, call corona_pud_classify — it does the full PUD pipeline in one shot.
- Cite the EXACT ICTV criterion and threshold you applied, and the EXACT computed value from your tool calls.
- Confidence levels: High (computed value clearly above/below threshold), Medium (within ±5% of threshold), Low (required computation failed or partial sequence).
- **Novel species heuristic for families WITHOUT numerical thresholds** (e.g. Astroviridae, Caliciviridae, Orthoherpesviridae, Pneumoviridae, Poxviridae, etc.): when get_criteria returns no numerical threshold, use these BLAST-based rules to judge novel_species:
    • blast_pident < 70% to the closest reference → almost certainly novel species (possibly novel genus). Set novel_species=true, confidence=Medium.
    • blast_pident 70-90% → likely novel species. Set novel_species=true, confidence=Low.
    • blast_pident > 90% → likely same species. Set novel_species=false, confidence=Medium.
    State clearly in reasoning that no official ICTV numerical threshold exists and the classification is a heuristic estimate based on sequence similarity.

## HMM-based region extraction
extract_target_region can extract family-specific target proteins from nucleotide genomes:
- Paramyxoviridae: L protein (~2200 aa)
- Flaviviridae: NS3, NS5 RdRp
- Rhabdoviridae: L protein, N protein
- Picornaviridae: P1 capsid, 3D polymerase
- Caliciviridae: VP1 capsid
- Papillomaviridae: L1 ORF
- Parvoviridae: NS1/Rep
- And 17 more families (Hantaviridae, Phenuiviridae, Nairoviridae, Peribunyaviridae, etc.)
Use this when the ICTV criteria specify a gene-region-specific comparison rather than whole-genome identity.

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
    client = _get_shared_client(api_key, base_url)

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
            f"Use this as a hint, but STILL call blast_and_compare first to obtain quantitative "
            f"global pairwise identity values — these are required for ICTV threshold comparison. "
            f"Then call get_criteria for {family_hint}."
        )

    user_message = (
        f"Please classify the following virus sequence.\n\n"
        f"Query ID: {query_id}\n"
        f"Sequence length: {len(raw_seq)} {'nt' if len(raw_seq) > 100 else 'aa'}\n\n"
        f"**The COMPLETE {len(raw_seq)}-character sequence is automatically injected into "
        f"every tool call by the system. You have full access to the entire sequence — "
        f"just call tools directly. Do NOT worry about sequence length or truncation. "
        f"The `sequence`/`genome_nt` fields are always auto-filled with ALL {len(raw_seq)} characters.**\n\n"
        f"Use the tools to identify the virus family and classify it according to ICTV criteria."
        f"{family_note}{avail_note}"
    )

    messages = [{"role": "user", "content": user_message}]

    # Store full sequence for tool use (injected into subsequent tool calls)
    _full_seq = raw_seq

    final_result: Optional[ClassifyResult] = None
    tool_result_store: list[tuple[str, str]] = []  # (tool_name, full_result_json)

    import asyncio
    import re as _re

    _in_think_block = False  # track <think> blocks that span multiple steps

    def _strip_think(text: str) -> tuple[str, bool]:
        """Remove <think>...</think> blocks, handling splits across steps.
        Returns (cleaned_text, still_inside_think_block)."""
        nonlocal _in_think_block
        if _in_think_block:
            # We're inside a think block from a previous step
            if '</think>' in text:
                text = text[text.index('</think>') + len('</think>'):]
                _in_think_block = False
            else:
                return '', True  # entire chunk is inside think block
        # Remove complete think blocks
        text = _re.sub(r'<think>.*?</think>', '', text, flags=_re.DOTALL)
        # Handle unclosed think block that started in this chunk
        if '<think>' in text:
            text = text[:text.index('<think>')]
            _in_think_block = True
        return text, _in_think_block

    for step in range(max_steps):
        # Run synchronous API call in thread pool to avoid blocking event loop
        # Disable extended thinking: GLM-4.7 otherwise generates huge thinking blocks
        # on complex classification prompts, making each round-trip take minutes.
        response = await asyncio.to_thread(
            client.messages.create,
            model=os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6"),
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
            thinking={"type": "disabled"},
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
                display_text, _ = _strip_think(combined_text)
                # Remove raw nucleotide sequences the model erroneously echoes
                display_text = _re.sub(
                    r'[ACGTUacgtuNn]{100,}',
                    lambda m: f'[sequence {len(m.group())} nt — truncated]',
                    display_text,
                )
                display_text = display_text.strip()
                if display_text:
                    await step_callback(f"[Step {step+1}] {display_text}")

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

                # Inject full query sequence so the model cannot truncate it.
                # For extract_target_region, skip the override when the model
                # is requesting a REFERENCE extraction (ref_accession set) —
                # otherwise we'd clobber the reference with the query.
                inputs = dict(tu.input)
                if tu.name in ("blast_search", "blast_and_compare"):
                    inputs["sequence"] = _full_seq
                elif tu.name in ("corona_pud_classify", "compare_query_to_reference"):
                    inputs["genome_nt"] = _full_seq
                elif tu.name == "extract_target_region":
                    if not inputs.get("ref_accession"):
                        inputs["genome_nt"] = _full_seq

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

    # Build a concise reasoning summary from structured data
    summary_parts = []
    if taxonomy.family:
        summary_parts.append(f"Identified as {taxonomy.family}")
    for e in evidence:
        if e.method == "global_pairwise_identity" and e.value is not None:
            summary_parts.append(f"global pairwise identity {e.value}% {e.conclusion}")
            break
    if taxonomy.species:
        summary_parts.append(f"closest species: {taxonomy.species}")
    reasoning = ". ".join(summary_parts) + "." if summary_parts else "See agent steps above."

    return ClassifyResult(
        query_id=query_id,
        taxonomy=taxonomy,
        confidence=confidence,
        novel_species=False,
        evidence=evidence,
        reasoning=reasoning,
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
