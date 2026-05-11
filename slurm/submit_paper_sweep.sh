#!/usr/bin/env bash
# Submit the 4-role paper-schedule sweep to Spartan.
#
#   placeholder (baseline, oracle_baseline) : n_epochs=3000, bs=1000
#     ⇒ 15000 SGD steps, matches upstream paper Figure 2 (nmmr_u_figure2.json)
#   MAR (oracle_modified, modified)        : n_epochs=3000, bs=5000 (full-batch)
#     ⇒ 3000 SGD steps, matches placeholder per-sample-view exposure (15M views)
#
# Per RESEARCH_DIARY.md 2026-05-10 tail-slope verdict, MAR plateaus at 2× of
# the Pass-4 schedule (= 300 epochs full-batch); 3000 epochs is comfortably
# past plateau. Placeholder reaches paper-grade convergence at 3000 epochs.
#
# Usage (from repo root, on Spartan login node):
#   bash slurm/submit_paper_sweep.sh
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs

# Placeholder roles: 8 CPUs, 6h walltime (Mac smoke 459s × 100 / 8 × 2x slowdown ≈ 3.2h)
for role in baseline oracle_baseline; do
    cfg="configs/mar_pci_configs/${role}_paper.json"
    if [[ ! -f "$cfg" ]]; then
        echo "missing config: $cfg" >&2
        exit 2
    fi
    tag="${role}_paper"
    echo "submitting: ${tag}  (placeholder, 8 CPUs, 6h)"
    SBATCH_TIMELIMIT=06:00:00 \
    CONFIG_PATH="$cfg" DUMP_TAG="$tag" \
        sbatch -J "nmmr_${tag}" slurm/run_one_config.sh
done

# MAR roles: 16 CPUs, 12h walltime (MAR full-batch + cross-fit is heavier;
# Mac smoke 801s × 100 / 16 × 2× slowdown ≈ 2.8h, but full-batch+folds may
# blow up — generous walltime as safety)
for role in oracle_modified modified; do
    cfg="configs/mar_pci_configs/${role}_paper.json"
    if [[ ! -f "$cfg" ]]; then
        echo "missing config: $cfg" >&2
        exit 2
    fi
    tag="${role}_paper"
    echo "submitting: ${tag}  (MAR, 16 CPUs, 12h)"
    SBATCH_TIMELIMIT=12:00:00 SBATCH_CPUS_PER_TASK=16 \
    CONFIG_PATH="$cfg" DUMP_TAG="$tag" \
        sbatch -J "nmmr_${tag}" slurm/run_one_config.sh
done

echo
echo "All 4 paper-schedule jobs submitted. Monitor with:"
echo "    squeue -u \$USER"
echo "Cancel all with:"
echo "    squeue -u \$USER -h -o '%i' -n nmmr_baseline_paper -n nmmr_oracle_baseline_paper -n nmmr_oracle_modified_paper -n nmmr_modified_paper | xargs -r scancel"
