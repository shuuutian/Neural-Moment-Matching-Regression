#!/usr/bin/env bash
# Submit all 12 (4 roles × 3 step multiples) convergence-sweep jobs to Spartan.
# Each job is independent and gets its own dump directory under
# dumps/nmmr_<timestamp>__<role>_<mult>x/.
#
# Usage (from repo root, on Spartan login node):
#   bash slurm/submit_convergence_sweep.sh
#
# The script intentionally does NOT use sbatch arrays — array indexing makes
# the (role, mult) decoding fragile when re-running a single config later.
set -euo pipefail

ROLES=(baseline oracle_baseline oracle_modified modified)
MULTS=(1 2 5)

cd "$(dirname "$0")/.."   # repo root
mkdir -p logs

for role in "${ROLES[@]}"; do
    for mult in "${MULTS[@]}"; do
        cfg="configs/mar_pci_configs/convergence_sweep/${role}_${mult}x.json"
        if [[ ! -f "$cfg" ]]; then
            echo "missing config: $cfg" >&2
            exit 2
        fi
        tag="${role}_${mult}x"
        echo "submitting: ${tag}  (cfg=${cfg})"
        CONFIG_PATH="$cfg" DUMP_TAG="$tag" \
            sbatch -J "nmmr_${tag}" slurm/run_one_config.sh
    done
done

echo
echo "All 12 jobs submitted. Monitor with:"
echo "    squeue -u \$USER"
echo "Cancel all with:"
echo "    squeue -u \$USER -h -o '%i' | xargs -r scancel"
