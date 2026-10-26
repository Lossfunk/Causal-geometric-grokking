#!/bin/bash
#SBATCH --job-name=fixedce
#SBATCH --partition=intern
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a5000:1
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --array=0-7
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err

# Transformer seeds 0-14 with the fixed cross-entropy, checkpoint every 250 steps.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR}"
source "${VENV_PATH:?pass the venv, e.g. sbatch --export=ALL,VENV_PATH=\$HOME/venvs/grok fixedce.sbatch}/bin/activate"
export PYTHONUNBUFFERED=1 MPLBACKEND=Agg OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

STATUS=0
for k in 0 1; do
    SEED=$(( SLURM_ARRAY_TASK_ID * 2 + k ))
    (( SEED > 14 )) && continue
    echo ">>> fixed CE, seed ${SEED}  ($(date))"
    python -m train.run_transformer --seeds "${SEED}" --wd 1.0 --ckpt-every 250 --loss fixed --out results_fixedce || {
        echo "WARNING: seed ${SEED} failed" >&2; STATUS=1; }
done
exit "$STATUS"