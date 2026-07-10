#!/usr/bin/env bash
# Section 6.5 — Sequential-training experiment (Hague dataset, 2 → 3 → 4 chains).
# Usage: bash scripts/section6_5_sequential_training.sh [--dry-run]

set -u

DRY_RUN=false
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "[dry-run] Commands will be printed but not executed."
fi

NUM_RUNS=5

BASE_CMD="python run_gnn.py \
    --embedding-branches-dir data/hague \
    --k-size 3 --edge-types both \
    --sequential"

total=0
passed=0
failed=0
failed_cmds=()

run_config() {
    local base_mods="$1"   # 2 modalities (trained jointly first)
    local add_mods="$2"    # 2 modalities, added one at a time in this order
    local label="[Hague | seq | base:(${base_mods}) → +${add_mods// / → +}]"
    local cmd="${BASE_CMD} --base-modalities ${base_mods} --add-modalities ${add_mods}"

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

# 12 chains = 6 base pairs × 2 orderings of the remaining pair.
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  2 → 3 → 4 SEQUENTIAL CHAINS                                      ║"
echo "╚══════════════════════════════════════════════════════════════════╝"

# base {image, feature_vector} | remaining {point_cloud, normals}
run_config "image feature_vector" "point_cloud normals"
run_config "image feature_vector" "normals point_cloud"

# base {image, point_cloud} | remaining {feature_vector, normals}
run_config "image point_cloud" "feature_vector normals"
run_config "image point_cloud" "normals feature_vector"

# base {image, normals} | remaining {feature_vector, point_cloud}
run_config "image normals" "feature_vector point_cloud"
run_config "image normals" "point_cloud feature_vector"

# base {feature_vector, point_cloud} | remaining {image, normals}
run_config "feature_vector point_cloud" "image normals"
run_config "feature_vector point_cloud" "normals image"

# base {feature_vector, normals} | remaining {image, point_cloud}
run_config "feature_vector normals" "image point_cloud"
run_config "feature_vector normals" "point_cloud image"

# base {point_cloud, normals} | remaining {image, feature_vector}
run_config "point_cloud normals" "image feature_vector"
run_config "point_cloud normals" "feature_vector image"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  SEQUENTIAL ABLATION COMPLETE"
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
