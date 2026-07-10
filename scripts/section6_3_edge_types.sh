#!/usr/bin/env bash
# Section 6.3 — Edge-type experiment.
# Compares graph connectivity: both / intra_only / inter_only, with k fixed to 3,
# Usage: bash scripts/section6_3_edge_types.sh [--dry-run]

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

EDGE_TYPES=(both intra_only inter_only)

total=0
passed=0
failed=0
failed_cmds=()

run_ablation() {
    local dataset="$1"
    local base_cmd="$2"
    local edges="$3"

    local label="[${dataset} | k=3 | edges=${edges}]"
    local cmd="${base_cmd} --k-size 3 --edge-types ${edges}"

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
echo "║  SECTION 6.3 — EDGE TYPES (both / intra_only / inter_only), k=3  ║"
echo "╚══════════════════════════════════════════════════════════════════╝"

for edges in "${EDGE_TYPES[@]}"; do
    run_ablation "Shoes"   "$BASE_SHOES"   "$edges"
    run_ablation "Zalando" "$BASE_ZALANDO" "$edges"
    run_ablation "Hague"   "$BASE_HAGUE"   "$edges"
done

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  SECTION 6.3 COMPLETE"
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
