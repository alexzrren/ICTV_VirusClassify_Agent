#!/usr/bin/env bash
# Per-family benchmark driver.
# Runs each family's test set serially through batch_classify.py with
# --parallel 6 (within-family concurrency). Output goes to benchmark/<family>/.
#
# Usage:
#   bash scripts/run_benchmark.sh

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
TEST_DIR="/home/renzirui/Database/NCBI_Viruses/test_dataset"
OUT_ROOT="$PROJECT_DIR/benchmark"
API="http://localhost:18231"
PARALLEL=6

mkdir -p "$OUT_ROOT"

FAMILIES=(
    Arenaviridae
    Astroviridae
    Coronaviridae
    Flaviviridae
    Hepeviridae
    Papillomaviridae
    Parvoviridae
    Picornaviridae
    Sedoreoviridae
)

# Sanity: server up?
if ! curl -s "$API/health" > /dev/null 2>&1; then
    echo "ERROR: ICTV Agent server is not running at $API" >&2
    echo "Start it: cd $PROJECT_DIR && bash run.sh" >&2
    exit 1
fi

echo "=== ICTV Agent Benchmark ==="
echo "Test dataset:  $TEST_DIR"
echo "Output root:   $OUT_ROOT"
echo "API:           $API"
echo "Parallel/fam:  $PARALLEL"
echo "Families:      ${FAMILIES[*]}"
echo ""

T_START=$(date +%s)
SUMMARY="$OUT_ROOT/_benchmark_summary.tsv"
echo -e "Family\tSequences\tDone\tErrors\tTotal_s\tAvg_s\tStarted_at" > "$SUMMARY"

for FAM in "${FAMILIES[@]}"; do
    FASTA="$TEST_DIR/${FAM}_testsequences.sampled.fna"
    if [ ! -f "$FASTA" ]; then
        echo ">>> SKIP $FAM (no fasta found at $FASTA)" >&2
        continue
    fi
    OUT_DIR="$OUT_ROOT/$FAM"
    mkdir -p "$OUT_DIR"

    N_SEQ=$(grep -c "^>" "$FASTA")
    NOW=$(date +%H:%M:%S)
    echo ""
    echo ">>> [$NOW] Starting $FAM ($N_SEQ sequences) → $OUT_DIR"

    T0=$(date +%s)
    # No --family hint: we want to test family identification too.
    # Long timeout (1200s) for difficult sequences.
    python3 "$PROJECT_DIR/scripts/batch_classify.py" \
        "$FASTA" \
        -o "$OUT_DIR" \
        --api "$API" \
        --parallel "$PARALLEL" \
        --timeout 1200 \
        2>&1 | tee -a "$OUT_DIR/run.log"
    T1=$(date +%s)
    ELAPSED=$((T1 - T0))

    # Extract done/errors from the log
    DONE=$(grep -oE "Done:[[:space:]]+[0-9]+" "$OUT_DIR/run.log" | tail -1 | grep -oE "[0-9]+" || echo "?")
    ERRORS=$(grep -oE "Errors:[[:space:]]+[0-9]+" "$OUT_DIR/run.log" | tail -1 | grep -oE "[0-9]+" || echo "?")
    AVG=$((ELAPSED / (N_SEQ > 0 ? N_SEQ : 1)))

    echo -e "$FAM\t$N_SEQ\t$DONE\t$ERRORS\t$ELAPSED\t$AVG\t$NOW" >> "$SUMMARY"
    echo ">>> [$FAM] done in ${ELAPSED}s. Done=$DONE, Errors=$ERRORS"
done

T_END=$(date +%s)
TOTAL=$((T_END - T_START))
echo ""
echo "================================="
echo "BENCHMARK COMPLETE"
echo "Total wall time: ${TOTAL}s ($(date -u -d @$TOTAL +%H:%M:%S))"
echo "Summary written: $SUMMARY"
echo "================================="
column -t -s $'\t' "$SUMMARY"
