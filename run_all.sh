#!/bin/bash
# =============================================================================
# run_all.sh -- train and analyze every run in the paper, on one machine or as
# a Slurm array job.
#
# On one machine:
#   ./run_all.sh --list                    # every run, grouped into suites
#   ./run_all.sh --parallel 4              # train everything, 4 runs at a time
#   ./run_all.sh --analyze                 # all analyses, after training
#
# On a Slurm cluster (add --partition, --account, a GPU type etc. on the sbatch
# command line if your cluster needs them):
#   mkdir -p logs                          # once: Slurm opens --output itself
#   ./run_all.sh --suite mlp --per-task 10 --list      # prints the --array range
#   sbatch --array=0-N run_all.sh --suite mlp --per-task 10
#   sbatch --array=0-N run_all.sh --suite mlp-cont --per-task 10   # after "mlp"
#   sbatch --array=0-N run_all.sh --suite transformer,holdout,dense,caps --per-task 10
#   sbatch run_all.sh --analyze            # after all training has finished
#
# Suites
#   mlp          all MLP jobs of run_mlp.py except the continuations
#   mlp-cont     optimizer continuations (cont_*), which start from the main
#                MLP runs at step 20,000 and therefore need suite "mlp" first
#   transformer  seeds 0-14 at weight decay 1.0, seeds 0-2 at 0.1 and 0.3
#   holdout      held-out seeds 15-29                      -> <out>_holdout
#   dense        the same runs with a checkpoint every 250 steps -> <out>_dense
#   caps         rank caps W = A B on W_in and W_E, r in {128,32,24,16,8}, seeds 0-4
# =============================================================================

#SBATCH --job-name=subspace_crystallization
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# ----------------------------------------------------------------- repo root
# Under sbatch the script runs from a spool directory, so look for the folder
# that actually contains the experiment scripts.
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
    echo "ERROR: could not find the folder with run_mlp.py and run_transformer.py." >&2
    echo "       Run from inside it, or set GROK_ROOT=/path/to/the/repository." >&2
    exit 2
fi
cd "$ROOT"

# ----------------------------------------------------------------- arguments
SUITES="all"; PER_TASK=1; PARALLEL=1; OUT="results"; ONLY=""; SKIP=""
VENV=""; CONDA_ENV=""; PY_OVERRIDE=""; ANALYZE=0; LIST=0; DRY=0

usage() {
    cat <<'USAGE'
Usage: run_all.sh [options]
  --suite LIST       comma-separated suites: all, mlp, mlp-cont, transformer,
                     holdout, dense, caps (default: all)
  --only REGEX       keep only runs whose description matches REGEX
  --skip REGEX       drop runs whose description matches REGEX
  --parallel N       runs to execute at the same time (default: 1)
  --per-task N       runs per Slurm array task (default: 1)
  --out DIR          results folder (default: results); held-out and dense
                     runs go to DIR_holdout and DIR_dense
  --analyze          run the analysis scripts instead of training
  --list             print every run, its array task and the --array range
  --dry-run          print the commands without running them
  --venv NAME|PATH   virtualenv; bare names are looked up in $WORKON_HOME,
                     ~/venvs, ~/.venvs, ~/.virtualenvs, ~/envs and ~/
  --conda NAME       conda/mamba environment
  --python PATH      use this interpreter directly
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --suite)    SUITES="$2"; shift 2 ;;
        --only)     ONLY="$2"; shift 2 ;;
        --skip)     SKIP="$2"; shift 2 ;;
        --parallel) PARALLEL="$2"; shift 2 ;;
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
for n in "$PER_TASK" "$PARALLEL"; do
    if ! [[ "$n" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: --per-task and --parallel expect a positive integer, got '$n'" >&2; exit 2
    fi
done
[[ "$SUITES" == "all" ]] && SUITES="mlp,mlp-cont,transformer,holdout,dense,caps"
IFS=',' read -r -a SUITE_LIST <<<"$SUITES"
for s in "${SUITE_LIST[@]}"; do
    if ! [[ "$s" =~ ^(mlp|mlp-cont|transformer|holdout|dense|caps)$ ]]; then
        echo "ERROR: unknown suite '$s'" >&2; usage >&2; exit 2
    fi
done
HOLDOUT="${OUT}_holdout"
DENSE="${OUT}_dense"

# ----------------------------------------------------------------- environment
# Never creates or installs anything, because concurrent array tasks would race.
if [[ -n "$CONDA_ENV" && -n "$VENV" ]]; then
    echo "ERROR: pass only one of --conda / --venv" >&2; exit 2
fi
ENV_LABEL="current shell"
if [[ -n "$CONDA_ENV" ]]; then
    CONDA_EXE_PATH="$(command -v conda || command -v mamba || command -v micromamba || true)"
    if [[ -z "$CONDA_EXE_PATH" ]]; then
        echo "ERROR: --conda given but no conda/mamba on PATH (on a cluster, you may need to load a module first)." >&2
        exit 2
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
        echo "ERROR: no virtualenv found for '$VENV'. Pass its full path." >&2; exit 2
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

# Fail early and readably if the environment lacks a package.
need_modules() {
    [[ "$DRY" -eq 1 || "$LIST" -eq 1 ]] && return 0
    local missing
    missing="$("$PY" -c 'import importlib.util as u, sys; print(",".join(m for m in sys.argv[1:] if u.find_spec(m) is None))' "$@" 2>/dev/null || echo PYTHON_FAILED)"
    if [[ "$missing" == "PYTHON_FAILED" ]]; then
        echo "ERROR: could not run '$PY'" >&2; exit 2
    fi
    if [[ -n "$missing" ]]; then
        echo "ERROR: environment '$ENV_LABEL' is missing: $missing" >&2
        echo "       Install it once, e.g.  $PY -m pip install ${missing//,/ }" >&2
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
    echo "Subspace crystallization experiments"
    echo "  started    : $(date)"
    echo "  host       : $(hostname)"
    echo "  folder     : $ROOT"
    echo "  python     : $PY ($("$PY" --version 2>&1))"
    echo "  environment: $ENV_LABEL"
    echo "  results    : $OUT, $HOLDOUT, $DENSE"
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        echo "  slurm job  : $SLURM_JOB_ID (array task ${SLURM_ARRAY_TASK_ID:-none})"
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
    else
        echo "  no nvidia-smi found: the scripts will use the CPU unless a GPU is visible to PyTorch"
    fi
    echo "============================================================"
}

# ----------------------------------------------------------------- analysis
if [[ "$ANALYZE" -eq 1 ]]; then
    need_modules statsmodels
    [[ "$DRY" -eq 0 ]] && banner
    STATUS=0
    have() { compgen -G "$1" >/dev/null; }
    run_cmd "$PY" analyze_mlp.py --runs "$OUT" || STATUS=$?
    if have "$OUT/transformer/wd*/seed*.pt"; then
        need_modules transformer_lens
        run_cmd "$PY" analyze_transformer.py --runs "$OUT" || STATUS=$?
        run_cmd "$PY" analyze_revision.py --runs "$OUT" --holdout "$HOLDOUT" --dense "$DENSE" || STATUS=$?
    else
        echo "skip: no transformer runs in $OUT, so analyze_transformer.py and analyze_revision.py are not run"
    fi
    if have "$OUT/mlp/cont_*/seed*.pt"; then
        run_cmd "$PY" analyze_mechanics.py --runs "$OUT" || STATUS=$?
    fi
    for task in mul p113; do
        if have "$OUT/mlp/${task}_main/seed*.pt"; then
            run_cmd "$PY" analyze_task.py --runs "$OUT" --task "$task" || STATUS=$?
        fi
    done
    if have "$HOLDOUT/transformer/wd*/seed*.pt"; then
        run_cmd "$PY" prereg_transformer_timing.py --runs "$HOLDOUT" || STATUS=$?
    fi
    exit "$STATUS"
fi

# ----------------------------------------------------------------- run list
# Each run is "suite|kind|fields...": mlp runs carry the job index of
# run_mlp.py --list, transformer runs carry seed, weight decay, output folder,
# checkpoint interval and rank cap ("-" when unused).
MLP_LIST="$("$PY" run_mlp.py --list)"
RUNS=()
add_tf() {   # suite seeds wd out ckpt factor
    local s
    for s in $(seq "${2%-*}" "${2#*-}"); do RUNS+=("$1|tf|$s|$3|$4|$5|$6"); done
}
for suite in "${SUITE_LIST[@]}"; do
    case "$suite" in
        mlp|mlp-cont)
            while read -r idx name seed; do
                [[ "$idx" == "#" ]] && continue
                if [[ "$name" == cont_* ]]; then
                    [[ "$suite" == "mlp-cont" ]] && RUNS+=("$suite|mlp|$idx")
                else
                    [[ "$suite" == "mlp" ]] && RUNS+=("$suite|mlp|$idx")
                fi
            done <<<"$MLP_LIST" ;;
        transformer)
            add_tf transformer 0-14 1.0 "$OUT" - -
            add_tf transformer 0-2 0.1 "$OUT" - -
            add_tf transformer 0-2 0.3 "$OUT" - - ;;
        holdout)
            add_tf holdout 15-29 1.0 "$HOLDOUT" - - ;;
        dense)
            add_tf dense 0-29 1.0 "$DENSE" 250 -
            add_tf dense 0-2 0.1 "$DENSE" 250 -
            add_tf dense 0-2 0.3 "$DENSE" 250 - ;;
        caps)
            for m in W_in W_E; do
                for r in 128 32 24 16 8; do add_tf caps 0-4 1.0 "$OUT" - "$m:$r"; done
            done ;;
    esac
done

describe() {
    local f
    IFS='|' read -r -a f <<<"$1"
    if [[ "${f[1]}" == "mlp" ]]; then
        printf 'mlp %s' "$(awk -v i="${f[2]}" '$1==i {print $2" seed "$3}' <<<"$MLP_LIST")"
    else
        printf 'transformer seed %s wd %s' "${f[2]}" "${f[3]}"
        [[ "${f[0]}" != "transformer" ]] && printf ' %s' "${f[0]}"
        [[ "${f[6]}" != "-" ]] && printf ' %s' "${f[6]}"
    fi
    return 0
}

if [[ -n "$ONLY" || -n "$SKIP" ]]; then
    KEPT=()
    for j in "${RUNS[@]}"; do
        d="$(describe "$j")"
        [[ -n "$ONLY" && ! "$d" =~ $ONLY ]] && continue
        [[ -n "$SKIP" && "$d" =~ $SKIP ]] && continue
        KEPT+=("$j")
    done
    RUNS=("${KEPT[@]+"${KEPT[@]}"}")
fi
N=${#RUNS[@]}
if (( N == 0 )); then echo "No runs selected." >&2; exit 2; fi
N_TASKS=$(( (N + PER_TASK - 1) / PER_TASK ))

if [[ "$LIST" -eq 1 ]]; then
    for (( j = 0; j < N; j++ )); do
        printf 'task %3d  run %3d  %-11s  %s\n' $(( j / PER_TASK )) "$j" "${RUNS[$j]%%|*}" "$(describe "${RUNS[$j]}")"
    done
    echo
    echo "$N runs at $PER_TASK per task -> $N_TASKS array tasks:"
    echo "  sbatch --array=0-$(( N_TASKS - 1 )) run_all.sh --suite $SUITES --per-task $PER_TASK${ONLY:+ --only '$ONLY'}${SKIP:+ --skip '$SKIP'}"
    if [[ ",$SUITES," == *",mlp,"* && ",$SUITES," == *",mlp-cont,"* ]]; then
        echo "  Under Slurm, submit 'mlp' and 'mlp-cont' separately: the continuations need the"
        echo "  main MLP runs, so start them after the 'mlp' array has finished (or use --dependency)."
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
    MINE=("${RUNS[@]:$(( SLURM_ARRAY_TASK_ID * PER_TASK )):PER_TASK}")
else
    MINE=("${RUNS[@]}")
fi
for j in "${MINE[@]}"; do
    if [[ "$j" == *"|tf|"* ]]; then need_modules transformer_lens; break; fi
done

command_for() {   # prints the training command of one run, one argument per line
    local f
    IFS='|' read -r -a f <<<"$1"
    if [[ "${f[1]}" == "mlp" ]]; then
        printf '%s\n' "$PY" run_mlp.py --job-index "${f[2]}" --out "$OUT"
    else
        printf '%s\n' "$PY" run_transformer.py --seeds "${f[2]}" --wd "${f[3]}" --out "${f[4]}"
        [[ "${f[5]}" != "-" ]] && printf '%s\n' --ckpt-every "${f[5]}"
        [[ "${f[6]}" != "-" ]] && printf '%s\n' --factor "${f[6]}"
    fi
    return 0
}

[[ "$DRY" -eq 0 ]] && banner && mkdir -p logs "$OUT"
STATUS=0
PIDS=()
finish_all() {   # wait for every background run and record failures
    local p
    for p in "${PIDS[@]+"${PIDS[@]}"}"; do wait "$p" || STATUS=1; done
    PIDS=()
}

PREV_SUITE=""
for j in "${MINE[@]}"; do
    suite="${j%%|*}"
    # The continuations need the main MLP runs, so finish one suite before the next.
    if [[ -n "$PREV_SUITE" && "$suite" != "$PREV_SUITE" ]]; then finish_all; fi
    PREV_SUITE="$suite"
    mapfile -t CMD < <(command_for "$j")
    desc="$(describe "$j")"
    if [[ "$DRY" -eq 1 ]]; then
        run_cmd "${CMD[@]}"
    elif (( PARALLEL == 1 )); then
        echo; echo "############ $desc ############"
        start=$SECONDS
        set +e; run_cmd "${CMD[@]}"; rc=$?; set -e
        echo "-- finished in $(( SECONDS - start ))s with exit code $rc"
        if [[ "$rc" -ne 0 ]]; then
            echo "WARNING: $desc failed; continuing with the next run." >&2
            STATUS=$rc
        fi
    else
        while (( ${#PIDS[@]} >= PARALLEL )); do
            wait "${PIDS[0]}" || STATUS=1
            PIDS=("${PIDS[@]:1}")
        done
        log="logs/$(tr ' :' '__' <<<"$desc").log"
        echo "start: $desc  (log: $log)"
        "${CMD[@]}" >"$log" 2>&1 &
        PIDS+=("$!")
    fi
done
finish_all
[[ "$DRY" -eq 0 ]] && echo && echo "All runs of this task finished (exit status $STATUS)."
exit "$STATUS"
