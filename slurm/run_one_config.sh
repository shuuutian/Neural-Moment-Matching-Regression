#!/bin/bash -l
#SBATCH -J nmmr_conv
#SBATCH -A punim2738
#SBATCH -p sapphire
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
# Override defaults via environment vars from the submitter:
#   SBATCH_TIMELIMIT=06:00:00      (longer wall for 5x MAR jobs)
#   SBATCH_CPUS_PER_TASK=16        (halve wall by doubling parallelism)
# These are picked up automatically by sbatch when set in the environment.
set -euo pipefail

# Driver expects two env vars:
#   CONFIG_PATH=configs/mar_pci_configs/convergence_sweep/<role>_<mult>x.json
#   DUMP_TAG=<role>_<mult>x
# Set by submit_convergence_sweep.sh; can also be exported manually for a
# one-off rerun (e.g. ``CONFIG_PATH=... DUMP_TAG=... sbatch slurm/run_one_config.sh``).

if [[ -z "${CONFIG_PATH:-}" || -z "${DUMP_TAG:-}" ]]; then
    echo "ERROR: CONFIG_PATH and DUMP_TAG must be set." >&2
    exit 2
fi

export PATH="$HOME/.local/bin:$PATH"

# Cap BLAS threads per worker. Without this, src.experiment uses
# os.cpu_count() (= 128 on sapphire nodes) // num_cpus (= 8) = 16 threads per
# worker × 8 workers = 128 threads on an 8-CPU SLURM allocation → heavy
# oversubscription. Pinning to 1 makes each worker single-threaded; the
# 8 reps still run in parallel via ProcessPoolExecutor.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs dumps

echo "=========================================="
echo "Job ID:      ${SLURM_JOB_ID}"
echo "Job name:    ${SLURM_JOB_NAME}"
echo "Node:        $(hostname)"
echo "CPUs:        ${SLURM_CPUS_PER_TASK}"
echo "Started at:  $(date)"
echo "Config:      ${CONFIG_PATH}"
echo "Dump tag:    ${DUMP_TAG}"
echo "=========================================="

uv run python main.py --dump-tag "${DUMP_TAG}" "${CONFIG_PATH}" ate -t "${SLURM_CPUS_PER_TASK}"

echo "=========================================="
echo "Finished at: $(date)"
echo "=========================================="
