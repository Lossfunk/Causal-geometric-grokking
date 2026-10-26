#!/bin/bash
#SBATCH --job-name=ce_wd
#SBATCH --partition=intern
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a5000:1
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --array=0-9
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# 20 runs (weight decay 0.03, 0.1, 0.3, 1 x seeds 0-4), 2 per array task: 10 tasks.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
source "${VENV_PATH:?pass the venv, e.g. sbatch --export=ALL,VENV_PATH=\$HOME/venvs/grok ce_wd.sbatch}/bin/activate"
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

WDS=(0.03 0.1 0.3 1)
STATUS=0
for k in 0 1; do
    i=$(( SLURM_ARRAY_TASK_ID * 2 + k ))
    WD=${WDS[$(( i / 5 ))]}
    SEED=$(( i % 5 ))
    echo ">>> ce_lr0.01_wd${WD} seed ${SEED}  ($(date))"
    python -m train.run_mlp --exp "ce_lr0.01_wd${WD}" --seeds "${SEED}" --out results || {
        echo "WARNING: ce_lr0.01_wd${WD} seed ${SEED} failed" >&2; STATUS=1; }
done
exit "$STATUS"