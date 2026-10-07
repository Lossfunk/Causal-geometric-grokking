# Subspace Crystallization (Revised experimental setup)

Every number and figure in the paper should come from the outputs of these
scripts: `summary.txt` / `summary.json` and the `fig_*.png` files. Nothing is
typed in by hand.

## Setup

```bash
pip install torch numpy scipy matplotlib statsmodels transformer_lens
```

`statsmodels` is only needed for the Granger section. `transformer_lens` is
only needed for the transformer scripts.

## Configurations (one per architecture)

| | MLP (`common_mlp.py`) | Transformer (`run_transformer.py`) |
|---|---|---|
| loss | **MSE** on one-hot targets | cross-entropy |
| optimizer | AdamW, lr 1e-3, wd 1e-2 | AdamW, lr 1e-3, wd 1.0, betas (0.9, 0.98) |
| train split | 35% | 30% |
| steps | 100,000 full batch | 25,000 full batch |
| logging | every 500 steps | every 250 steps |
| seeds | 0–14 | 0–14 |


## Run order

### 1. MLP training (≈ 5 GPU-hours total, 101 jobs, all independent)

```bash
python run_mlp.py --list            # shows every job: index, experiment, seed
python run_mlp.py --exp main        # M1: 15 seeds   (run this first)
python run_mlp.py --exp random_labels   # M3: negative control + memorizing models
python run_mlp.py --exp loss_ce         # M5
python run_mlp.py --exp act_relu ; python run_mlp.py --exp act_gelu      # M7
python run_mlp.py --exp wd_0 ; python run_mlp.py --exp wd_0.001          # M6 (wd=1e-2 is "main")
python run_mlp.py --exp wd_0.05 ; python run_mlp.py --exp wd_0.1
python run_mlp.py --exp nuc_1e-08   # M4a, also nuc_3e-08 nuc_1e-07 nuc_3e-07 nuc_1e-06
python run_mlp.py --exp rank_8      # M4b, also rank_16 rank_32 rank_48 rank_64 rank_80 rank_96
```

The same jobs can run as a Slurm array (the cluster allows 10 jobs at a time):

```bash
#!/bin/bash
#SBATCH --array=0-100%10
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
python run_mlp.py --job-index $SLURM_ARRAY_TASK_ID --out results
```

### 2. Transformer training (≈ 1.5 GPU-hours)

```bash
python run_transformer.py --seeds 0-14            # T1
python run_transformer.py --seeds 0-2 --wd 0.1    # T3
python run_transformer.py --seeds 0-2 --wd 0.3    # T3
```

### 3. Analyses (minutes; rerun any time more results arrive)

```bash
python analyze_mlp.py --runs results
python analyze_transformer.py --runs results
python continual_mlp.py --runs results --seeds 0-2     # optional
```

## Where each paper item comes from

| Paper item | Source |
|---|---|
| Fig. 1 training dynamics | `analysis_mlp/fig_training_dynamics.png` (seed 0) |
| Stage boundaries / Stage-4 ER drop | `analysis_mlp/summary.txt` §A, §J |
| Fig. 2 weight-decay sweep, "onset vs wd" claim | §J [weight decay], `fig_wd_sweep.png` |
| Stage 4 under MSE vs CE  | §J [loss], `fig_loss_comparison.png` |
| §2.1 ReLU/GELU claim | §J [activation], `fig_activation_comparison.png` |
| Fig. 3a, App. B (causal dims at 4 thresholds) | §E, `fig_causal_dimension.png` |
| Table 2, MLP row (top/bottom/random) | §E |
| Complement interventions (keep C, keep C⊥, noise in C vs C⊥) | §E |
| App. C subspace stability (now including pre-grokking steps) | §F |
| Activation patching statement | §H |
| Transplant + Procrustes, App. D | §I |
| Table 3, Fig. 7a/b, lead–lag statistics | §A, §B, `fig_leadlag_*.png` |
| Granger analysis (with stationarity check) | §C |
| IPR claims, onset sensitivity, label-free comparison, false alarms | §D |
| Rank intervention: penalty from step 0, hard bottleneck | §J [nuclear-norm], [bottleneck] |
| §5 overlap, Fig. 5 | §G, `fig_fourier_spectra_seeds.png` |
| Fig. 9b/c PCA, App. G phase + robustness | §K figures |
| Table 6 (d_c of every transformer matrix), attention paragraph | `analysis_transformer` §B |
| Table 2, transformer rows | `analysis_transformer` §B/C |
| Fig. 8 spectra, k95/k99 (variance) | `analysis_transformer` §A, `fig_transformer_spectra.png` |
| Read/write ("matched filter") alignment + correct baseline | `analysis_transformer` §D, §I |
| Transformer lead–lag (Table 1 "not analysed" row) | `analysis_transformer` §E |
| Table 4 IPR sensitivity | `analysis_transformer` §F (and MLP §D) |
| Fig. 9a logit PCA | `fig_logit_pca_transformer.png` |
| "Free capacity" test (optional) | `analysis_continual/summary.txt` |

## Smoke test

```bash
python run_mlp.py --exp main --seeds 0-1 --steps 600 --log-every 100 --out smoke_outputs
python analyze_mlp.py --runs smoke_outputs
```

600 steps is far too short to grok. The smoke test only checks that the code
runs end to end.
