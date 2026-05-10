#!/usr/bin/env bash
# Generate the full convergence-sweep visualization grid from the 12 dumps.
#
# Inputs: 12 dumps named dumps/nmmr_<timestamp>__<role>_<mult>x for the
# canonical {baseline, oracle_baseline, oracle_modified, modified} × {1, 2, 5}
# grid.
#
# Outputs:
#   plots/convergence/per_a_box_<mult>x.png   — 4-role per-a bias boxplots,
#                                               one per multiple (the same
#                                               view as plots/oos/ but at
#                                               each step budget)
#   plots/convergence/training_<role>.png     — per-role training-history
#                                               overlay (1×/2×/5× lines)
#
# Usage (from repo root, after rsync'ing the 12 dumps back from Spartan):
#   bash scripts/run_convergence_analysis.sh
set -euo pipefail

cd "$(dirname "$0")/.."

ROLES=(baseline oracle_baseline oracle_modified modified)
MULTS=(1 2 5)

# Resolve dump-dir for a given (role, mult). Picks the most recent matching
# folder if there are multiple.
find_dump() {
    local tag="$1"
    local d
    d=$(ls -td "dumps/nmmr_"*"__${tag}" 2>/dev/null | head -1 || true)
    if [[ -z "$d" ]]; then
        echo "ERROR: no dump found for tag ${tag}" >&2
        return 1
    fi
    echo "$d"
}

mkdir -p plots/convergence

# 1) Per-multiple per-a bias boxplots: 4 roles overlaid in each figure.
for mult in "${MULTS[@]}"; do
    args=()
    for role in "${ROLES[@]}"; do
        d=$(find_dump "${role}_${mult}x")
        args+=("$d")
    done
    echo "=== per-a boxplot at ${mult}x ==="
    uv run python scripts/plot_validation.py \
        "${args[@]}" \
        --out "plots/convergence/per_a_${mult}x" \
        --title-suffix "  step multiple ${mult}×  test∈[10, 30]"
done

# 2) Per-role training history: 1×/2×/5× overlaid for each role.
# plot_validation.py uses path::label to disambiguate same-role dumps; the
# training-history script reuses that mechanism.
for role in "${ROLES[@]}"; do
    args=()
    for mult in "${MULTS[@]}"; do
        d=$(find_dump "${role}_${mult}x")
        args+=("${d}::${role}_${mult}x")
    done
    echo "=== training history for ${role} ==="
    uv run python scripts/plot_training_history.py \
        "${args[@]}" \
        --out "plots/convergence/training_${role}.png" \
        --title-suffix "  ${role}  (1× / 2× / 5× step multiples)"
done

echo
echo "Wrote plots to plots/convergence/"
