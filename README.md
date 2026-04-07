# ICTV Virus Classification Agent

An LLM-powered agent that classifies virus sequences by strictly following official **ICTV demarcation criteria**, rather than phylogenetic placement alone.

> **详细技术文档**：[ICTV病毒分类智能Agent实施方案.md](ICTV病毒分类智能Agent实施方案.md)

## Overview

- Reads ICTV Report Chapters for each family to extract official demarcation standards
- Calls bioinformatics tools (BLAST, MAFFT, HMMER) to compute the metrics each family requires
- Compares computed values against ICTV thresholds
- Returns a classification with full evidence chain (method, region, value, threshold)

Supported: **32 virus families**, ICTV MSL40 (16,213 species).

## Quick Start

```bash
# Requirements: blast+, mafft, hmmer (via conda/micromamba)
pip install -r requirements.txt

export ANTHROPIC_API_KEY=your-key
export ANTHROPIC_BASE_URL=https://your-endpoint  # optional
export CLAUDE_MODEL=claude-sonnet-4-6             # optional

bash run.sh         # default port 18231
```

Open **http://localhost:18231**.

## Key Capabilities

| Feature | Status |
|---------|--------|
| 32 virus families, ReAct Agent (9 tools) | ✅ |
| 21 families with numerical thresholds | ✅ |
| 98 genera with genus-level criteria | ✅ |
| Coronaviridae subgenus (DEmARC PUD, 5 domains, validated SARS-CoV-1/2) | ✅ |
| MSL40 taxonomy DB (16,213 species) | ✅ |
| TF-IDF search over 32 ICTV report chapters | ✅ |
| SSE real-time reasoning stream | ✅ |
| Other families' reference sequences + BLAST DB | ⚠️ partial |

## Agent Tools

1. `blast_and_compare` — BLAST + global pairwise identity in one call
2. `get_criteria` — ICTV demarcation rules for a family
3. `lookup_taxonomy` — full taxonomy from MSL40
4. `search_ictv_docs` — TF-IDF search in ICTV report chapters
5. `compute_pairwise_identity` — MAFFT alignment
6. `fetch_reference_sequence` — reference sequence by accession
7. `list_reference_species` — MSL40 species list
8. `blast_search` — standalone BLAST
9. `corona_pud_classify` — DEmARC PUD subgenus classification (Coronaviridae)

## API

```bash
# Submit classification
curl -X POST http://localhost:18231/classify \
  -H 'Content-Type: application/json' \
  -d '{"fasta": ">query\nATGCGA...", "family_hint": "Coronaviridae"}'

# Stream reasoning (SSE)
curl http://localhost:18231/stream/{job_id}

# Get result
curl http://localhost:18231/result/{job_id}

# Family criteria
curl http://localhost:18231/family/Coronaviridae
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes | LLM API key |
| `ANTHROPIC_BASE_URL` | No | Custom endpoint (Anthropic-compatible providers) |
| `CLAUDE_MODEL` | No | Model name (default: `claude-sonnet-4-6`) |
| `BLASTN_DB` | No | BLAST database path prefix |

## Comparison with EPA-ng

| | EPA-ng | This Agent |
|---|---|---|
| Classification basis | Phylogenetic nearest-neighbor | ICTV official thresholds |
| Explainability | Low | High (evidence + thresholds cited) |
| Speed | Fast | 30–120s per query |
| Best use | High-throughput screening | Standard-compliant formal classification |

## License

MIT
