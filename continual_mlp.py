"""
continual_mlp.py -- OPTIONAL: is the complement of the sufficient subspace usable as free capacity for a second task?

Task 1 = (a+b) mod p : a trained "main" run (W1, readout W2 frozen).
Task 2 = (a-b) mod p : a new readout head, trained with MSE for --steps.
W1 is updated in three ways:
  unconstrained  W1 trained freely
  protect_C      every W1 update is projected so U_k U_k^T W1 stays fixed,
                 i.e. the C-component of the pre-activations is unchanged for
                 all inputs (k = d_0.99 of the run)
  frozen         W1 not trained (only the new head)
Reports task-1 test accuracy (retention) and task-2 test accuracy.
Note: with a quadratic activation, changing the C-perp component of h still
changes z = h^2 in every unit, so protect_C need not preserve task 1 -- that
is exactly what this test measures.

  python continual_mlp.py --runs results --seeds 0-2
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn

from analyze_mlp import d_at, topk_curve
from common_mlp import (P, WIDTH, accuracy, compute_loss, final_model, load_run, make_dataset, run_data)


def parse_seeds(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out.extend(range(int(a), int(b or a) + 1))
    return out


def run_variant(run, variant, k, steps, lr, wd, device, log_every):
    X, y1, _, te1 = run_data(run, device)
    model = final_model(run, device)
    _, y2 = make_dataset(P, op="sub")
    y2 = y2.to(device)
    g = torch.Generator().manual_seed(1000 + run["seed"])
    perm = torch.randperm(P * P, generator=g)
    n_tr = int(0.35 * P * P)
    tr2, te2 = perm[:n_tr].to(device), perm[n_tr:].to(device)
    head = nn.Parameter(torch.randn(P, WIDTH, generator=g).to(device))
    params = [head] + ([] if variant == "frozen" else [model.W1_full])
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    U = torch.linalg.svd(model.W1_full.detach(), full_matrices=False)[0][:, :k]
    Pc = U @ U.T

    def evaluate():
        with torch.no_grad():
            z = model.act(model.pre(X))
            a1 = accuracy(model.readout(z[te1]), y1[te1])
            a2 = accuracy((head @ z[te2].T).T / WIDTH, y2[te2])
        return a1, a2

    curve = [(0, *evaluate())]
    for step in range(1, steps + 1):
        W_before = model.W1_full.detach().clone()
        z = model.act(model.pre(X[tr2]))
        loss = compute_loss((head @ z.T).T / WIDTH, y2[tr2], "mse")
        opt.zero_grad()
        loss.backward()
        opt.step()
        if variant == "protect_C":
            with torch.no_grad():
                d = model.W1_full - W_before
                model.W1_full.copy_(W_before + d - Pc @ d)
        if step % log_every == 0 or step == steps:
            curve.append((step, *evaluate()))
    return curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results")
    ap.add_argument("--seeds", default="0-2")
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--log-every", type=int, default=1000)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    out = args.out or os.path.join(args.runs, "analysis_continual")
    os.makedirs(out, exist_ok=True)
    results, lines = {}, []
    for seed in parse_seeds(args.seeds):
        run = load_run(os.path.join(args.runs, "mlp", "main", f"seed{seed}.pt"))
        X, y, _, te = run_data(run)
        with torch.no_grad():
            ks, accs, orig, _ = topk_curve(final_model(run), X[te], y[te])
        k = d_at(ks, accs, orig, 0.99) or len(ks)
        for variant in ("unconstrained", "protect_C", "frozen"):
            curve = run_variant(run, variant, k, args.steps, args.lr, args.wd, args.device, args.log_every)
            results[f"seed{seed}|{variant}"] = curve
            s, a1, a2 = curve[-1]
            lines.append(f"seed {seed}  k={k:>3}  {variant:<14} task-1 retained {100 * a1:6.2f}%   "
                         f"task-2 learned {100 * a2:6.2f}%   (task-1 before: {100 * curve[0][1]:.2f}%)")
            print(lines[-1], flush=True)
    summary = {}
    for variant in ("unconstrained", "protect_C", "frozen"):
        a1 = [results[k][-1][1] for k in results if k.endswith(variant)]
        a2 = [results[k][-1][2] for k in results if k.endswith(variant)]
        summary[variant] = (float(np.mean(a1)), float(np.mean(a2)))
        lines.append(f"MEAN {variant:<14} task-1 {100 * np.mean(a1):.2f}%  task-2 {100 * np.mean(a2):.2f}%")
    with open(os.path.join(out, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"curves": results, "summary": summary}, f, indent=1)
    print("\n".join(lines[-3:]))


if __name__ == "__main__":
    main()
