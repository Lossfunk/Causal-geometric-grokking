"""
run_mlp.py -- train the mean-field MLP for every experiment in the paper.

Examples
  python run_mlp.py --exp main --seeds 0-14          # M1  (15 seeds)
  python run_mlp.py --exp random_labels --seeds 0-4  # M3  IPR negative control / memorizing models
  python run_mlp.py --exp nuc_1e-07 --seeds 0-4      # M4a rank penalty from step 0
  python run_mlp.py --exp rank_32 --seeds 0-2        # M4b hard rank-32 bottleneck
  python run_mlp.py --exp rank_128 --seeds 0-2       # M4b control: factorised W1 = A B at full rank
  python run_mlp.py --exp loss_ce_lr0.01             # M5  cross-entropy, lr 1e-2 (also loss_ce_lr0.003)
  python run_mlp.py --exp loss_ce --seeds 0-4        # M5  cross-entropy instead of MSE
  python run_mlp.py --exp wd_0 --seeds 0-4           # M6  weight-decay sweep
  python run_mlp.py --exp act_relu --seeds 0-4       # M7  activation comparison

  python run_mlp.py --list                 # print every (exp, seed) job, one per line
  python run_mlp.py --job-index 17         # run job #17 of that list (Slurm arrays)

Each run is saved to <out>/mlp/<exp>/seed<k>.pt and skipped if it exists.
"""
import argparse
import os
import time

import torch

from common_mlp import DEFAULTS, train_run

# experiment name -> (config overrides, default seeds)
EXPERIMENTS = {
    "main":          ({}, range(15)),
    "random_labels": ({"labels": "random"}, range(5)),
    "loss_ce":       ({"loss": "ce"}, range(5)),
    "act_relu":      ({"act": "relu"}, range(5)),
    "act_gelu":      ({"act": "gelu"}, range(5)),
}
for wd in [0.0, 1e-3, 5e-2, 1e-1]:                 # wd = 1e-2 is "main"
    EXPERIMENTS[f"wd_{wd:g}"] = ({"wd": wd}, range(5))
for lam in [1e-8, 3e-8, 1e-7, 3e-7, 1e-6]:         # nuclear-norm penalty on W1 from step 0
    EXPERIMENTS[f"nuc_{lam:.0e}"] = ({"lam_nuc": lam, "nuc_start": 0}, range(5))
for r in [8, 16, 32, 48, 64, 80, 96]:               # hard bottleneck W1 = A B (full rank = "main")
    EXPERIMENTS[f"rank_{r}"] = ({"rank": r}, range(3))
# Added for the final version. Appended last so the job indices above stay unchanged.
EXPERIMENTS["rank_128"] = ({"rank": 128}, range(3))  # W1 = A B at full rank: separates factorisation from rank
for lr in [3e-3, 1e-2]:                             # cross-entropy with larger learning rates
    EXPERIMENTS[f"loss_ce_lr{lr:g}"] = ({"loss": "ce", "lr": lr}, range(5))
# Second tasks (PREREGISTRATION.md, H2-H6): multiplication mod 97 and addition mod 113.
# Same configuration as "main" otherwise. Appended last so earlier job indices stay unchanged.
for tag, task in [("mul", {"op": "mul"}), ("p113", {"p": 113})]:
    EXPERIMENTS[f"{tag}_main"] = (dict(task), range(15))
    for r in [128, 64, 32]:                         # factorised control (128) and two rank limits
        EXPERIMENTS[f"{tag}_rank_{r}"] = ({**task, "rank": r}, range(3))


def parse_seeds(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def all_jobs():
    return [(name, seed) for name, (_, seeds) in EXPERIMENTS.items() for seed in seeds]


def run_one(name, seed, args, device):
    overrides, _ = EXPERIMENTS[name]
    cfg = dict(overrides)
    if args.steps:
        cfg["steps"] = args.steps
    if args.log_every:
        cfg["log_every"] = args.log_every
    path = os.path.join(args.out, "mlp", name, f"seed{seed}.pt")
    if os.path.exists(path) and not args.overwrite:
        print(f"[skip] {path} exists")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"[run] exp={name} seed={seed} cfg={ {**DEFAULTS, **cfg} }", flush=True)
    t0 = time.time()
    run = train_run(seed, cfg, device=device)
    run["exp"] = name
    torch.save(run, path)
    h = run["hist"]
    print(f"[done] {path}  final test_acc={h['test_acc'][-1]:.4f}  "
          f"ER(W1)={h['W1_erank'][-1]:.1f}  ({time.time() - t0:.0f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=sorted(EXPERIMENTS))
    ap.add_argument("--seeds", type=str, default=None,
                    help="e.g. 0-14 or 0,3,5 (default: the experiment's standard seeds)")
    ap.add_argument("--steps", type=int, default=None, help="override (smoke tests)")
    ap.add_argument("--log-every", type=int, default=None, help="override (smoke tests)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--list", action="store_true", help="print all jobs and exit")
    ap.add_argument("--job-index", type=int, default=None)
    args = ap.parse_args()

    jobs = all_jobs()
    if args.list:
        for i, (name, seed) in enumerate(jobs):
            print(i, name, seed)
        print(f"# {len(jobs)} jobs")
        return
    if args.job_index is not None:
        name, seed = jobs[args.job_index]
        run_one(name, seed, args, args.device)
        return
    if not args.exp:
        ap.error("give --exp, --list or --job-index")
    seeds = parse_seeds(args.seeds) if args.seeds else list(EXPERIMENTS[args.exp][1])
    for seed in seeds:
        run_one(args.exp, seed, args, args.device)


if __name__ == "__main__":
    main()
