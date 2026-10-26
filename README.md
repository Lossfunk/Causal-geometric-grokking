# Subspace Crystallization in Grokking

Code for *Subspace Crystallization in Grokking: Rank Compression Tracks the Phase Transition*
(Shlok Mehendale, Paras Chopra · Lossfunk). Preprint: arXiv link coming soon.

Every number and figure in the paper comes from the outputs of these scripts: the `summary.txt` / `summary.json` files and the `fig_*.png` files in the `analysis_*` folders, which are included here.

## Layout

```
grokking/        shared MLP code (model, data, training loop, effective rank, onset detection)
train/           run_mlp.py, run_transformer.py (also the fixed cross-entropy, --loss fixed)
analysis/        one script per analysis; each writes summary.txt, summary.json and figures
slurm/           Slurm job files for the weight-decay sweep and the fixed cross-entropy reruns
run_all.sh       trains and analyses everything, locally or as a Slurm array
results*/        analysis outputs (raw runs and checkpoints are not included)
```

Run every command from the repository root. Scripts are modules, e.g. `python -m analysis.analyze_mlp`.

## Setup

```bash
pip install -r requirements.txt
```

`transformer_lens` must be 3.x: version 4.0 removed `HookedTransformer`. `statsmodels` is only needed for the Granger tests.

## Configurations

| | MLP (`grokking/common_mlp.py`, `train/run_mlp.py`) | Transformer (`train/run_transformer.py`) |
|---|---|---|
| Task | (a+b) mod 97, plus ab mod 97 and (a+b) mod 113 | (a+b) mod 97 |
| Loss | mean-squared error on one-hot targets | cross-entropy (`--loss fixed` for the fixed version) |
| Optimizer | AdamW, lr 1e-3, weight decay 1e-2, eps 1e-8 | AdamW, lr 1e-3, weight decay 1.0, betas (0.9, 0.98) |
| Train split | 35% | 30% |
| Steps | 100,000, full batch | 25,000, full batch |
| Logging | every 500 steps | every 250 steps |
| Seeds | 0–14 | 0–14, and 15–29 held out for the replication |

## 1. Training

The quickest way to run everything is `run_all.sh`:

```bash
./run_all.sh --list                 # every run, grouped into suites
./run_all.sh --parallel 4           # train everything on one machine, 4 runs at a time
./run_all.sh --analyze              # all analyses, after training
```

It also runs as a Slurm array job (see the header of `run_all.sh`). The commands below do the same thing step by step.

### MLP (261 independent jobs)

```bash
python -m train.run_mlp --list                 # every job: index, experiment, seed
python -m train.run_mlp --exp main             # 15 seeds, run this first
python -m train.run_mlp --job-index 0 --out results
```

The jobs cover the main runs, random-label controls, cross-entropy, weight decay, ReLU/GELU, nuclear-norm penalties, rank-constrained `W1 = A B` (`rank_*`), the two further tasks (`mul_*`, `p113_*`), runs logged every 50 steps (`fine_*`), and optimizer continuations from step 20,000 of the main runs (`cont_*`). Run `main` before the `cont_*` jobs, because they start from its checkpoints.

Cross-entropy MLPs at the transformer's per-step weight decay (Section 8, Appendix B):

```bash
sbatch --export=ALL,VENV_PATH=$HOME/venvs/grok slurm/ce_wd.sbatch
```

### Transformer

```bash
python -m train.run_transformer --seeds 0-14 --out results                  # main runs
python -m train.run_transformer --seeds 0-2 --wd 0.1 --out results          # negative controls
python -m train.run_transformer --seeds 0-2 --wd 0.3 --out results
python -m train.run_transformer --seeds 15-29 --out results_holdout         # held-out seeds
python -m train.run_transformer --seeds 0-29 --ckpt-every 250 --out results_dense     # dense checkpoints
python -m train.run_transformer --seeds 0-2 --wd 0.1 --ckpt-every 250 --out results_dense
python -m train.run_transformer --seeds 0-2 --wd 0.3 --ckpt-every 250 --out results_dense
```

Rank caps write `W = A B` with inner dimension r and save to `results/transformer_cap/<matrix>_r<r>/`:

```bash
for m in W_in W_E; do for r in 128 32 24 16 8; do
  python -m train.run_transformer --seeds 0-4 --factor $m:$r --out results
done; done
```

Reruns with the fixed cross-entropy of Barron et al. (2026), which removes softmax collapse:

```bash
python -m train.run_transformer --seeds 0-14 --ckpt-every 250 --loss fixed --out results_fixedce
# or: sbatch --export=ALL,VENV_PATH=$HOME/venvs/grok slurm/fixedce.sbatch
```

The Slurm files use our cluster's partition and GPU type. Change the `#SBATCH` lines for yours.

## 2. Analyses

```bash
python -m analysis.analyze_mlp --runs results                     # -> results/analysis_mlp
python -m analysis.analyze_transformer --runs results             # -> results/analysis_transformer
python -m analysis.analyze_revision --runs results --holdout results_holdout --dense results_dense   # -> results/analysis_revision
python -m analysis.analyze_mechanics --runs results               # -> results/analysis_mechanics
python -m analysis.analyze_task --runs results --task mul         # -> results/analysis_mul
python -m analysis.analyze_task --runs results --task p113        # -> results/analysis_p113
python -m analysis.prereg_transformer_timing --runs results_holdout   # -> results_holdout/prereg_H1
python -m analysis.analyze_ce_wd --runs results                   # -> results/analysis_ce_wd
python -m analysis.check_softmax_collapse --runs results --holdout results_holdout   # -> results/analysis_softmax_collapse
python -m analysis.analyze_revision --runs results_fixedce --holdout none --dense results_fixedce --sections A,D   # -> results_fixedce/analysis_revision
```

`analysis.analyze_revision --sections A,B` runs a subset. Each section heading in `summary.txt` names what it computes.

## Where each paper item comes from

| Paper item | Source |
|---|---|
| Teaser (a): steepest effective-rank fall vs onset and vs fit | `analysis_revision` §A, `fig_tf_ts_vs_tg_tfit.png` |
| Teaser (b) and the dense transformer table | `analysis_revision` §D, `fig_tf_dense_margin_free.png` |
| Four training stages of the MLP | `analysis_mlp` §A, §K, `fig_rank_derivatives.png` |
| Timing table, lead-lag figures | `analysis_mlp` §A, §B (MLP), `analysis_revision` §A (transformer) |
| Matched-step table | `analysis_revision` §A2 |
| Runs that never grok, random-label MLPs | `analysis_revision` §A, §B, `fig_mlp_er_random_labels.png` |
| Granger tests | `analysis_mlp` §C |
| IPR onset detection | `analysis_mlp` §D, `analysis_transformer` §F |
| Causal dimensionality, truncation controls, complement interventions | `analysis_mlp` §E, `fig_causal_dimension.png` |
| Sufficient subspace over training, margin-free counts (MLP) | `analysis_mlp` §E2, §E4, `analysis_revision` §C |
| Stage-4 widening | `analysis_mlp` §E3, `fig_stage4_widening.png` |
| Temporal stability of the subspace | `analysis_mlp` §F |
| Equivalent solutions across seeds (overlap, transplants, CKA) | `analysis_mlp` §G, §I, `analysis_revision` §E |
| Training variants (weight decay, loss, activation) | `analysis_mlp` §J and its `fig_*_comparison.png`, `fig_wd_sweep.png` |
| Rank constraints and nuclear-norm penalty (MLP) | `analysis_mlp` §J, `analysis_revision` §G |
| Rank caps (transformer) | `analysis_revision` §H |
| Adam state in Stage 4 | `analysis_revision` §F |
| Optimizer swap | `analysis_mechanics` §A |
| Transformer dimensionality, spectra, controls | `analysis_transformer` §A–C |
| Residual-stream alignment | `analysis_transformer` §D, §I |
| Logit geometry, frequencies, non-grokked runs | `analysis_transformer` §H, §J, §N |
| Replication on held-out seeds and tasks | `results_holdout/prereg_H1`, `analysis_mul`, `analysis_p113` |
| Synthetic check of the steepest-drop measure | `analysis_revision` §I |
| Weight decay and the transformer–MLP gap (Section 8, Appendix B) | `analysis_ce_wd` |
| Softmax collapse in the transformer runs | `analysis_softmax_collapse` |
| Results with the fixed cross-entropy | `results_fixedce/analysis_revision` §A, §D |

## Smoke test

```bash
python -m train.run_mlp --exp main --seeds 0-1 --steps 600 --log-every 100 --out smoke_outputs
python -m analysis.analyze_mlp --runs smoke_outputs
```

600 steps is far too short to grok. The smoke test only checks that the code runs end to end.

## License

MIT (see `LICENSE`).
