#!/usr/bin/env bash
# Section 6.2 — Neighborhood-size (k) experiment.
# Varies the number of intra-modality k-NN neighbors (k = 1..6) with edge-types
# Usage: bash scripts/section6_2_neighborhood_size.sh [--dry-run]

set -u

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "[dry-run] Commands will be printed but not executed."
fi

NUM_RUNS=5

BASE_SHOES="python run_gnn.py --embedding-branches-dir data/shoes"
BASE_ZALANDO="python run_gnn.py --embedding-branches-dir data/zalando"
BASE_HAGUE="python run_gnn.py --embedding-branches-dir data/hague"

K_SIZES=(1 2 3 4 5 6)

total=0
passed=0
failed=0
failed_cmds=()

run_ablation() {
    local dataset="$1"
    local base_cmd="$2"
    local k="$3"

    local label="[${dataset} | k=${k} | edges=both]"
    local cmd="${base_cmd} --k-size ${k} --edge-types both"

    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  ${label}"
    echo "════════════════════════════════════════════════════════════════"
    echo "  CMD: ${cmd}"
    echo ""

    for run_idx in $(seq 1 ${NUM_RUNS}); do
        local run_label="${label} run ${run_idx}/${NUM_RUNS}"
        echo "  ── ${run_label}"
        total=$((total + 1))

        if $DRY_RUN; then
            echo "     [dry-run] skipping execution"
            continue
        fi

        if eval "${cmd}"; then
            passed=$((passed + 1))
            echo "  ✓ ${run_label} completed"
        else
            failed=$((failed + 1))
            failed_cmds+=("${run_label}")
            echo "  ✗ ${run_label} FAILED — continuing..."
        fi
    done
}

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  SECTION 6.2 — NEIGHBORHOOD SIZE (k), edge-types=both            ║"
echo "╚══════════════════════════════════════════════════════════════════╝"

for k in "${K_SIZES[@]}"; do
    run_ablation "Shoes"   "$BASE_SHOES"   "$k"
    run_ablation "Zalando" "$BASE_ZALANDO" "$k"
    run_ablation "Hague"   "$BASE_HAGUE"   "$k"
done

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  SECTION 6.2 COMPLETE"
if ! $DRY_RUN; then
    echo "  Total runs : ${total}"
    echo "  Passed     : ${passed}"
    echo "  Failed     : ${failed}"
    if [[ ${#failed_cmds[@]} -gt 0 ]]; then
        echo "  Failed runs:"
        for f in "${failed_cmds[@]}"; do
            echo "    - ${f}"
        done
    fi
fi
echo "════════════════════════════════════════════════════════════════"
