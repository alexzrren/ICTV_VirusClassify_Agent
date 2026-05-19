#!/usr/bin/env bash
# Benchmark multiple LLM models on the same FASTA test set.
# Usage:
#   export ANTHROPIC_API_KEY="<volcano-ark-key>"
#   bash scripts/benchmark_models.sh <test.fasta> <output_root> <model1> [<model2> ...]
#
# Example:
#   bash scripts/benchmark_models.sh \
#     /path/dty_CoVs.fasta \
#     /path/benchmark_results \
#     glm-4.7 glm-5.1 kimi-k2.6
#
# Each model runs in its own output subdir.
# Assumes Volcano Ark base URL by default; override via ANTHROPIC_BASE_URL.

set -e

if [ $# -lt 3 ]; then
    echo "Usage: $0 <input.fasta> <output_root> <model1> [<model2> ...]" >&2
    exit 1
fi

FASTA="$1"; shift
OUT_ROOT="$1"; shift
MODELS=("$@")

if [ -z "${ANTHROPIC_API_KEY:-}" ]; then
    echo "ERROR: ANTHROPIC_API_KEY not set (Volcano Ark key)" >&2
    exit 1
fi

export ANTHROPIC_BASE_URL="${ANTHROPIC_BASE_URL:-https://ark.cn-beijing.volces.com/api/coding}"
PORT="${PORT:-18231}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

mkdir -p "$OUT_ROOT"
echo "=== Benchmark ==="
echo "FASTA:     $FASTA ($(grep -c ">" "$FASTA") sequences)"
echo "Output:    $OUT_ROOT"
echo "Models:    ${MODELS[*]}"
echo "Base URL:  $ANTHROPIC_BASE_URL"
echo ""

for MODEL in "${MODELS[@]}"; do
    OUT_DIR="$OUT_ROOT/$MODEL"
    mkdir -p "$OUT_DIR"
    echo ">>> Testing model: $MODEL (out: $OUT_DIR)"

    # Clear cache between models so every run is fresh
    python3 "$PROJECT_DIR/scripts/clear_cache.py" || true

    # Start server in background with this model
    (cd "$PROJECT_DIR" && \
      CLAUDE_MODEL="$MODEL" \
      ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY" \
      ANTHROPIC_BASE_URL="$ANTHROPIC_BASE_URL" \
      python3 -m uvicorn backend.main:app --host 0.0.0.0 --port "$PORT" --log-level warning \
      > "$OUT_DIR/server.log" 2>&1) &
    SERVER_PID=$!

    # Wait for server to be ready
    for i in $(seq 1 30); do
        if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
            echo "    Server ready (pid=$SERVER_PID, model=$MODEL)"
            break
        fi
        sleep 1
    done
    if ! curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "    ERROR: Server failed to start. See $OUT_DIR/server.log"
        kill $SERVER_PID 2>/dev/null || true
        continue
    fi

    # Run batch classification
    python3 "$PROJECT_DIR/scripts/batch_classify.py" \
        "$FASTA" \
        -o "$OUT_DIR" \
        --api "http://localhost:$PORT" \
        --parallel 3 \
        --family Coronaviridae

    # Stop server
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    sleep 2
    echo "    Done. Stopped server."
    echo ""
done

echo "=== All models tested ==="
echo "Summary (rows done/error + avg time + avg tokens per Excel):"
for MODEL in "${MODELS[@]}"; do
    XLSX="$OUT_ROOT/$MODEL/results_summary.xlsx"
    if [ -f "$XLSX" ]; then
        echo "  $MODEL:  $XLSX"
    else
        echo "  $MODEL:  MISSING (check $OUT_ROOT/$MODEL/server.log)"
    fi
done
