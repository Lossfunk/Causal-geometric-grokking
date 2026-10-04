#!/bin/bash
# =============================================================================
# Grokking paper -- experiment launcher (Slurm array or plain bash).
#
# 122 training runs = 101 MLP (python run_mlp.py --list) + 21 transformer
# (seeds 0-14 at wd 1.0; seeds 0-2 at wd 0.1 and 0.3). Every run skips itself
# if its .pt already exists, so resubmitting only fills in what is missing.
#
#   mkdir -p logs                                          # once, BEFORE sbatch
#   ./run_grokking.sh --venv grok --list --per-task 13     # runs + --array range
#   sbatch --array=0-9 run_grokking.sh --venv grok --per-task 13
#   sbatch run_grokking.sh --venv grok --analyze           # after all runs finish
#
# logs/ must exist before sbatch: Slurm opens --output itself, before the
# script body runs.
#
# Every array task counts against the per-user submit limit (10); %N only
# throttles how many run. --per-task 13 packs the 122 runs into 10 tasks.
# =============================================================================

#SBATCH --job-name=subspace_Crys
#SBATCH --partition=intern
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a5000:1
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# gpu:a5000:1 is required on `intern`: a bare gpu:1 can be placed on the V100
# node (not ours) and pend forever. The models are tiny, so 4 CPUs + 8 GB per
# task keeps ten concurrent tasks at 80 GB, under the per-user memory cap.

set -euo pipefail

# ----------------------------------------------------------------- repo root
# Under sbatch the script runs from Slurm's spool directory, so look for the
# folder that actually contains the experiment scripts.
find_root() {
    local c sd
    sd="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd || true)"
    for c in "${GROK_ROOT:-}" "${SLURM_SUBMIT_DIR:-}" "$sd" "$PWD"; do
        if [[ -n "$c" && -f "$c/run_mlp.py" && -f "$c/run_transformer.py" ]]; then
            printf '%s\n' "$c"
            return 0
        fi
    done
    return 1
}
if ! ROOT="$(find_root)"; then
    echo "ERROR: could not find the paper_experiments folder (run_mlp.py, run_transformer.py)." >&2
    echo "       Submit from inside it, or set GROK_ROOT=/path/to/paper_experiments." >&2
    exit 2
fi
cd "$ROOT"

# ----------------------------------------------------------------- arguments
SUITE="all"; PER_TASK=1; OUT="results"; VENV=""; CONDA_ENV=""; PY_OVERRIDE=""
ANALYZE=0; LIST=0; DRY=0

usage() {
    cat <<'USAGE'
Usage: run_grokking.sh [options]
  --suite all|mlp|transformer   which training runs (default: all)
  --per-task N                  runs per Slurm array task (default: 1)
  --out DIR                     results folder (default: results)
  --analyze                     run the analysis scripts instead of training
  --list                        print every run, its array task, and the --array range
  --dry-run                     print the commands without running them
  --venv NAME|PATH              virtualenv; bare names are looked up in
                                ~/venvs, ~/.venvs, ~/.virtualenvs, ~/envs, ~/, $WORKON_HOME
  --conda NAME                  conda/mamba environment
  --python PATH                 use this interpreter directly
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --suite)    SUITE="$2"; shift 2 ;;
        --per-task) PER_TASK="$2"; shift 2 ;;
        --out)      OUT="$2"; shift 2 ;;
        --venv)     VENV="$2"; shift 2 ;;
        --conda)    CONDA_ENV="$2"; shift 2 ;;
        --python)   PY_OVERRIDE="$2"; shift 2 ;;
        --analyze)  ANALYZE=1; shift ;;
        --list)     LIST=1; shift ;;
        --dry-run)  DRY=1; shift ;;
        -h|--help)  usage; exit 0 ;;
        *) echo "ERROR: unknown option '$1'" >&2; usage >&2; exit 2 ;;
    esac
done
if ! [[ "$SUITE" =~ ^(all|mlp|transformer)$ ]]; then
    echo "ERROR: --suite must be all, mlp or transformer" >&2; exit 2
fi
if ! [[ "$PER_TASK" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: --per-task expects a positive integer, got '$PER_TASK'" >&2; exit 2
fi

# ----------------------------------------------------------------- environment
# Never creates or installs anything: concurrent array tasks would race.
if [[ -n "$CONDA_ENV" && -n "$VENV" ]]; then
    echo "ERROR: pass only one of --conda / --venv" >&2; exit 2
fi
ENV_LABEL="ambient"
if [[ -n "$CONDA_ENV" ]]; then
    CONDA_EXE_PATH="$(command -v conda || command -v mamba || command -v micromamba || true)"
    if [[ -z "$CONDA_EXE_PATH" ]]; then
        echo "ERROR: --conda given but no conda/mamba on PATH (try 'module load anaconda')." >&2; exit 2
    fi
    CONDA_BASE="$("$CONDA_EXE_PATH" info --base 2>/dev/null || true)"
    set +u
    if [[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
        # shellcheck disable=SC1091
        source "$CONDA_BASE/etc/profile.d/conda.sh"
    fi
    if ! conda activate "$CONDA_ENV"; then
        echo "ERROR: could not activate conda environment '$CONDA_ENV'" >&2; exit 2
    fi
    set -u
    ENV_LABEL="conda:$CONDA_ENV"
elif [[ -n "$VENV" ]]; then
    if [[ "$VENV" != */* && "$VENV" != "~"* ]]; then
        for c in "${WORKON_HOME:-/nonexistent}/$VENV" "$HOME/venvs/$VENV" "$HOME/.venvs/$VENV" \
                 "$HOME/.virtualenvs/$VENV" "$HOME/envs/$VENV" "$HOME/$VENV"; do
            if [[ -f "$c/bin/activate" ]]; then VENV="$c"; break; fi
        done
    fi
    VENV="${VENV/#\~/$HOME}"
    if [[ ! -f "$VENV/bin/activate" ]]; then
        echo "ERROR: no virtualenv found for '$VENV'. Pass the full path, e.g. --venv \$HOME/venvs/grok" >&2
        exit 2
    fi
    set +u
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
    set -u
    ENV_LABEL="venv:$VENV"
fi

if [[ -n "$PY_OVERRIDE" ]]; then
    PY="$PY_OVERRIDE"; ENV_LABEL="python:$PY_OVERRIDE"
else
    PY="$(command -v python3 || command -v python || true)"
fi
if [[ -z "$PY" ]]; then echo "ERROR: no python interpreter found" >&2; exit 2; fi

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-${OMP_NUM_THREADS:-4}}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
mkdir -p logs "$OUT"

# Fail early and readably if the environment lacks a package.
need_modules() {
    local missing
    missing="$("$PY" -c 'import importlib.util as u, sys; print(",".join(m for m in sys.argv[1:] if u.find_spec(m) is None))' "$@" 2>/dev/null || echo PYTHON_FAILED)"
    if [[ "$missing" == "PYTHON_FAILED" ]]; then
        echo "ERROR: could not run '$PY'" >&2; exit 2
    fi
    if [[ -n "$missing" ]]; then
        echo "ERROR: environment '$ENV_LABEL' is missing: $missing" >&2
        echo "       Install once on the login node:  $PY -m pip install ${missing//,/ }" >&2
        exit 2
    fi
}
need_modules torch numpy scipy matplotlib

run_cmd() {
    if [[ "$DRY" -eq 1 ]]; then
        printf 'Would run: '; printf '%q ' "$@"; printf '\n'; return 0
    fi
    echo ">>> $*"
    "$@"
}

banner() {
    echo "============================================================"
    echo "Grokking experiments"
    echo "  started    : $(date)"
    echo "  host       : $(hostname)"
    echo "  folder     : $ROOT"
    echo "  python     : $PY ($("$PY" --version 2>&1))"
    echo "  environment: $ENV_LABEL"
    echo "  results    : $OUT"
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        echo "  slurm job  : $SLURM_JOB_ID (array task ${SLURM_ARRAY_TASK_ID:-none})"
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
    else
        echo "  nvidia-smi not found -- running on CPU"
    fi
    echo "============================================================"
}

# ----------------------------------------------------------------- analysis
if [[ "$ANALYZE" -eq 1 ]]; then
    need_modules statsmodels
    banner
    STATUS=0
    run_cmd "$PY" analyze_mlp.py --runs "$OUT" || STATUS=$?
    if compgen -G "$OUT/transformer/wd*/seed*.pt" >/dev/null; then
        need_modules transformer_lens
        run_cmd "$PY" analyze_transformer.py --runs "$OUT" || STATUS=$?
    fi
    run_cmd "$PY" continual_mlp.py --runs "$OUT" --seeds 0-2 || STATUS=$?
    exit "$STATUS"
fi

# ----------------------------------------------------------------- run list
JOBS=()
MLP_LIST=""
if [[ "$SUITE" != "transformer" ]]; then
    MLP_LIST="$("$PY" run_mlp.py --list)"
    N_MLP="$(awk '/^# /{print $2}' <<<"$MLP_LIST")"
    for (( i = 0; i < N_MLP; i++ )); do JOBS+=("mlp $i"); done
fi
if [[ "$SUITE" != "mlp" ]]; then
    for s in $(seq 0 14); do JOBS+=("tf $s 1.0"); done
    for wd in 0.1 0.3; do
        for s in 0 1 2; do JOBS+=("tf $s $wd"); done
    done
fi
N=${#JOBS[@]}
N_TASKS=$(( (N + PER_TASK - 1) / PER_TASK ))

describe() {
    local a
    read -r -a a <<<"$1"
    if [[ "${a[0]}" == "mlp" ]]; then
        printf 'mlp %s' "$(awk -v i="${a[1]}" '$1==i {print $2" seed "$3}' <<<"$MLP_LIST")"
    else
        printf 'transformer seed %s wd %s' "${a[1]}" "${a[2]}"
    fi
}

if [[ "$LIST" -eq 1 ]]; then
    for (( j = 0; j < N; j++ )); do
        printf 'task %3d  run %3d  %s\n' $(( j / PER_TASK )) "$j" "$(describe "${JOBS[$j]}")"
    done
    echo
    echo "$N runs at $PER_TASK per task -> $N_TASKS array tasks:"
    echo "  sbatch --array=0-$(( N_TASKS - 1 )) run_grokking.sh --suite $SUITE --per-task $PER_TASK --venv <env>"
    if (( N_TASKS > 10 )); then
        echo "  NOTE: more than 10 tasks exceeds the per-user submit limit; raise --per-task."
    fi
    exit 0
fi

# ----------------------------------------------------------------- this task's runs
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    if (( SLURM_ARRAY_TASK_ID >= N_TASKS )); then
        echo "ERROR: array index $SLURM_ARRAY_TASK_ID is past the $N runs at $PER_TASK per task." >&2
        echo "       Use --array=0-$(( N_TASKS - 1 ))." >&2
        exit 2
    fi
    START=$(( SLURM_ARRAY_TASK_ID * PER_TASK ))
    MINE=("${JOBS[@]:START:PER_TASK}")
else
    MINE=("${JOBS[@]}")
fi
for j in "${MINE[@]}"; do
    if [[ "$j" == tf* ]]; then need_modules transformer_lens; break; fi
done

banner
echo "Runs in this task:"
for j in "${MINE[@]}"; do echo "  $(describe "$j")"; done

STATUS=0
for j in "${MINE[@]}"; do
    read -r -a a <<<"$j"
    echo
    echo "############ $(describe "$j") ############"
    start=$SECONDS
    set +e
    if [[ "${a[0]}" == "mlp" ]]; then
        run_cmd "$PY" run_mlp.py --job-index "${a[1]}" --out "$OUT"
    else
        run_cmd "$PY" run_transformer.py --seeds "${a[1]}" --wd "${a[2]}" --out "$OUT"
    fi
    rc=$?
    set -e
    echo "-- finished in $(( SECONDS - start ))s with exit code $rc"
    if [[ "$rc" -ne 0 ]]; then
        echo "WARNING: $(describe "$j") failed; continuing with the next run." >&2
        STATUS=$rc
    fi
done
exit "$STATUS"