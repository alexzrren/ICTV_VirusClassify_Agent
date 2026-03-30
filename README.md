# ICTV Virus Classification Agent

An LLM-powered agent that classifies virus sequences by strictly following official **ICTV (International Committee on Taxonomy of Viruses) demarcation criteria**, rather than phylogenetic placement alone.

## Overview

Traditional methods like EPA-ng classify viruses by placing sequences on a reference tree (nearest-neighbor). This agent instead:

- Reads ICTV Report Chapters for each family to extract official demarcation standards
- Calls bioinformatics tools (BLAST, MAFFT, HMMER) to compute the metrics each family requires
- Compares computed values against the ICTV thresholds
- Returns a classification with full evidence chain (method used, region, computed value, threshold)

Supported: **32 virus families** from ICTV MSL40 (16,213 species).

## Features

| Feature | Description |
|---------|-------------|
| ReAct Agent | LLM reasons + calls tools in a loop (up to 20 steps) |
| Family-specific tools | Each family uses different regions/metrics per ICTV criteria |
| `blast_and_compare` | One-step BLAST + global pairwise identity calculation |
| MSL40 taxonomy DB | SQLite with 16,213 species, full Realm→Species hierarchy |
| TF-IDF document search | 32 ICTV Report Chapters indexed, 1,736 chunks |
| Web interface | FastAPI + SSE real-time reasoning stream |
| Multi-model support | Any Anthropic-compatible API (Claude, GLM-4.7, etc.) |
| Graceful degradation | Works without BLAST DB using `family_hint` |

## Quick Start

### Requirements

```bash
conda install -c bioconda blast mafft hmmer emboss
pip install -r requirements.txt
```

### Start Server

```bash
# Set credentials
export ANTHROPIC_API_KEY=your-key
export ANTHROPIC_BASE_URL=https://your-endpoint  # optional, for compatible providers
export CLAUDE_MODEL=claude-sonnet-4-6             # optional

# Start (default port 18231)
bash run.sh

# Stop
bash stop.sh
```

Open **http://localhost:18231** in your browser.

### Build Reference Database (optional)

Without a BLAST database, the agent still works using `family_hint` and criteria lookup. To enable full BLAST-based classification:

```bash
# Download reference sequences from NCBI
python scripts/download_reference_seqs.py \
    --vmr /path/to/VMR_MSL40.v2.20251013.xlsx \
    --families /path/to/vf.list \
    --email your@email.com

# Build BLAST database
python scripts/build_blast_db.py
```

## API

```bash
# Submit a classification job
curl -X POST http://localhost:18231/classify \
  -H 'Content-Type: application/json' \
  -d '{
    "fasta": ">query\nATGCGA...",
    "family_hint": "Coronaviridae",
    "max_steps": 8
  }'
# → {"job_id": "...", "status": "pending"}

# Poll result
curl http://localhost:18231/result/{job_id}

# Stream reasoning steps (Server-Sent Events)
curl http://localhost:18231/stream/{job_id}

# Query ICTV criteria for a family
curl http://localhost:18231/family/Coronaviridae

# List all supported families
curl http://localhost:18231/families
```

## Architecture

```
FASTA Input
    │
    ▼
┌───────────────────────────────────────────────────────────┐
│  LLM ReAct Agent  (Anthropic tool_use)                    │
│                                                           │
│  1. blast_and_compare  → BLAST + global pairwise identity │
│  2. get_criteria       → retrieve ICTV demarcation rules  │
│  3. lookup_taxonomy    → full taxonomy from MSL40         │
│  4. search_ictv_docs   → TF-IDF search in report chapters │
│  5. compute_pairwise_identity  → MAFFT alignment          │
│  6. fetch_reference_sequence   → get ref seq by accession │
│  7. list_reference_species     → MSL40 species list       │
│  8. blast_search       → standalone BLAST search          │
└───────────────────────────────────────────────────────────┘
    │
    ▼
ClassifyResult
  taxonomy:   Realm → Kingdom → … → Species
  confidence: High / Medium / Low
  novel:      true/false
  evidence:   [{method, region, value, threshold, conclusion}]
  reasoning:  step-by-step log
```

## Project Structure

```
ictv_agent/
├── backend/
│   ├── agent.py              # ReAct agent loop (tool_use)
│   ├── main.py               # FastAPI endpoints + SSE
│   ├── models.py             # Pydantic schemas
│   ├── tools/
│   │   ├── blast.py          # BLAST / DIAMOND wrappers
│   │   ├── alignment.py      # MAFFT pairwise identity
│   │   ├── hmmer.py          # HMMER region extraction
│   │   └── taxonomy.py       # SQLite MSL40 queries
│   └── knowledge/
│       ├── criteria.py       # criteria.json loader
│       └── rag.py            # TF-IDF search over ICTV chapters
├── frontend/
│   └── index.html            # Bootstrap 5 + SSE web UI
├── scripts/
│   ├── download_reference_seqs.py
│   ├── build_blast_db.py
│   ├── build_taxonomy_db.py
│   └── extract_criteria.py   # LLM-based criteria extraction
├── data/
│   ├── taxonomy.db           # MSL40 (16,213 species)
│   └── criteria.json         # Demarcation criteria (32 families)
├── run.sh
└── stop.sh
```

## Supported Families

32 virus families from ICTV MSL40, including:

`Coronaviridae` `Picornaviridae` `Paramyxoviridae` `Flaviviridae` `Caliciviridae`
`Rhabdoviridae` `Papillomaviridae` `Polyomaviridae` `Hantaviridae` `Filoviridae`
`Arenaviridae` `Adenoviridae` `Orthoherpesviridae` `Togaviridae` `Circoviridae`
`Parvoviridae` `Astroviridae` `Hepeviridae` `Nodaviridae` `Pneumoviridae`
`Bornaviridae` `Arteriviridae` `Asfarviridae` `Anelloviridae` `Nairoviridae`
`Peribunyaviridae` `Phenuiviridae` `Picobirnaviridae` `Poxviridae` `Sedoreoviridae`
`Spinareoviridae` `Amnoonviridae`

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | Yes | — | LLM API key |
| `ANTHROPIC_BASE_URL` | No | Anthropic default | Custom endpoint for compatible providers |
| `CLAUDE_MODEL` | No | `claude-sonnet-4-6` | Model name |
| `BLASTN_DB` | No | `data/db/blastn_ref` | BLAST database path prefix |
| `NCBI_EMAIL` | No | — | For NCBI sequence downloads |

## Comparison with EPA-ng

| | EPA-ng (phylogenetic placement) | This Agent (ICTV criteria) |
|---|---|---|
| Classification basis | Tree nearest-neighbor | Official ICTV thresholds |
| Explainability | Low (LCA result only) | High (evidence + thresholds cited) |
| Novel species detection | pendant length + LWR | Below-threshold flag + LLM reasoning |
| Speed | Fast (seconds) | Slower (30-120s per query) |
| Best use | High-throughput screening | Standard-compliant formal classification |

The two approaches are complementary: EPA-ng for fast initial screening, this agent for standard-based final classification.

## License

MIT
