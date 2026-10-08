# Subspace Crystallization in Grokking

Code for *Subspace Crystallization in Grokking: Rank Compression Tracks the Phase Transition* (Shlok Mehendale and Paras Chopra, Lossfunk).

Every number and figure in the paper comes from the outputs of these scripts: the `summary.txt` / `summary.json` files and the `fig_*.png` files in the `analysis_*` folders.

## Setup

```bash
pip install torch numpy scipy matplotlib statsmodels transformer_lens
```

`statsmodels` is only needed for the Granger tests, and `transformer_lens` only for the transformer scripts.

## Configurations

| | MLP (`common_mlp.py`, `run_mlp.py`) | Transformer (`run_transformer.py`) |
|---|---|---|
| Task | (a+b) mod 97, plus ab mod 97 and (a+b) mod 113 | (a+b) mod 97 |
| Loss | mean-squared error on one-hot targets | cross-entropy |
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

It also runs as a Slurm array job (see the header of `run_all.sh`). The individual commands below do the same thing step by step.

### MLP (261 independent jobs)

```bash
python run_mlp.py --list                 # every job: index, experiment, seed
python run_mlp.py --exp main             # 15 seeds, run this first
python run_mlp.py --job-index 0 --out results
```

The jobs cover the main runs, random-label controls, cross-entropy, weight decay, ReLU/GELU, nuclear-norm penalties, rank-constrained `W1 = A B` (`rank_*`), the two further tasks (`mul_*`, `p113_*`), runs logged every 50 steps (`fine_*`), and optimizer continuations from step 20,000 of the main runs (`cont_*`). As a Slurm array:

```bash
#!/bin/bash
#SBATCH --array=0-260%10
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
python run_mlp.py --job-index $SLURM_ARRAY_TASK_ID --out results
```

Run `main` before the `cont_*` jobs, because they start from its checkpoints.

### Transformer

```bash
python run_transformer.py --seeds 0-14 --out results                  # main runs
python run_transformer.py --seeds 0-2 --wd 0.1 --out results          # negative controls
python run_transformer.py --seeds 0-2 --wd 0.3 --out results
python run_transformer.py --seeds 15-29 --out results_holdout         # held-out seeds
python run_transformer.py --seeds 0-29 --ckpt-every 250 --out results_dense     # dense checkpoints
python run_transformer.py --seeds 0-2 --wd 0.1 --ckpt-every 250 --out results_dense
python run_transformer.py --seeds 0-2 --wd 0.3 --ckpt-every 250 --out results_dense
```

Rank caps write `W = A B` with inner dimension r and save to `results/transformer_cap/<matrix>_r<r>/`:

```bash
for m in W_in W_E; do for r in 128 32 24 16 8; do
  python run_transformer.py --seeds 0-4 --factor $m:$r --out results
done; done
```

## 2. Analyses

```bash
python analyze_mlp.py --runs results                     # -> results/analysis_mlp
python analyze_transformer.py --runs results             # -> results/analysis_transformer
python analyze_revision.py --runs results --holdout results_holdout --dense results_dense   # -> results/analysis_revision
python analyze_mechanics.py --runs results               # -> results/analysis_mechanics
python analyze_task.py --runs results --task mul         # -> results/analysis_mul
python analyze_task.py --runs results --task p113        # -> results/analysis_p113
python prereg_transformer_timing.py --runs results_holdout   # -> results_holdout/prereg_H1
```

`analyze_revision.py --sections A,B` runs a subset. Each section heading in `summary.txt` names what it computes.

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

The `analysis_*` folders with these outputs are included in this repository.

## Smoke test

```bash
python run_mlp.py --exp main --seeds 0-1 --steps 600 --log-every 100 --out smoke_outputs
python analyze_mlp.py --runs smoke_outputs
```

600 steps is far too short to grok. The smoke test only checks that the code runs end to end.
