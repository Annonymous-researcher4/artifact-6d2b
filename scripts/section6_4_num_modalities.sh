#!/usr/bin/env bash
# Section 6.4 — Number-of-modalities experiment (Hague dataset).
# Runs all 2-modality and 3-modality combinations.
# Usage: bash scripts/section6_4_num_modalities.sh [--dry-run]

set -u

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "[dry-run] Commands will be printed but not executed."
fi

NUM_RUNS=5

BASE_CMD="python run_gnn.py \
    --embedding-branches-dir data/hague \
    --k-size 3 --edge-types both"

total=0
passed=0
failed=0
failed_cmds=()

run_config() {
    local modalities="$1"
    local label="[Hague | modalities: ${modalities}]"
    local cmd="${BASE_CMD} --modalities ${modalities}"

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

# ── 2-modality combinations  C(4,2) = 6 ─────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  2-MODALITY COMBINATIONS                                         ║"
echo "╚══════════════════════════════════════════════════════════════════╝"

run_config "image feature_vector"
run_config "image point_cloud"
run_config "image normals"
run_config "feature_vector point_cloud"
run_config "feature_vector normals"
run_config "point_cloud normals"

# ── 3-modality combinations  C(4,3) = 4 ─────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  3-MODALITY COMBINATIONS                                         ║"
echo "╚══════════════════════════════════════════════════════════════════╝"

run_config "image feature_vector point_cloud"
run_config "image feature_vector normals"
run_config "image point_cloud normals"
run_config "feature_vector point_cloud normals"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  MODALITY ABLATION COMPLETE"
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
