#!/usr/bin/env bash
# Start the ICTV Classifier web server.
# Usage: bash run.sh [port]
#
# Required env vars:
#   ANTHROPIC_API_KEY   — Claude API key
#
# Optional env vars:
#   BLASTN_DB           — path to blastn DB prefix (default: data/db/blastn_ref)
#   DIAMOND_DB          — path to diamond .dmnd file (default: data/db/diamond_ref.dmnd)
#   HMM_DIR             — path to HMM profiles dir
#   NCBI_EMAIL          — for sequence download scripts
#   CLAUDE_MODEL        — override Claude model (default: claude-sonnet-4-6)
#   ANTHROPIC_BASE_URL  — custom API base URL (for compatible providers)

set -e

PORT=${1:-18231}

# Default LLM config (volcano engine GLM-4.7, Anthropic-compatible)
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-sk-e01b2ba657c93505b189544dc5b214e850dec89c1d80aafda5c5e80e07c041eb}"
export ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:-https://api.svips.org}"
export CLAUDE_MODEL="${CLAUDE_MODEL:-MiniMax-M2.7-highspeed}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Activate micromamba base environment if available
if [ -f "$HOME/micromamba/etc/profile.d/mamba.sh" ]; then
    source "$HOME/micromamba/etc/profile.d/mamba.sh"
    micromamba activate base 2>/dev/null || true
fi

cd "$SCRIPT_DIR"

echo "=== ICTV Virus Classifier ==="
echo "Port: $PORT"
echo "API key: ${ANTHROPIC_API_KEY:+SET}${ANTHROPIC_API_KEY:-NOT SET}"
echo ""

if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "WARNING: ANTHROPIC_API_KEY not set — LLM classification will fail."
fi

python3 -m uvicorn backend.main:app --host 0.0.0.0 --port "$PORT" --reload
