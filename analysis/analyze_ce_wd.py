"""
analyze_ce_wd.py -- does stronger weight decay make the cross-entropy MLP crystallize?

  python -m analysis.analyze_ce_wd --runs results

Reads results/mlp/loss_ce_lr0.01 (wd 1e-2) and results/mlp/ce_lr0.01_wd*/ and writes
<runs>/analysis_ce_wd/summary.txt. Per weight decay: seeds that grok, t_g, final test accuracy,
final effective rank of W1, and d_out / d_0.99 of W1 (test inputs, centered logits, as in
Table tab:margin_free) at the first checkpoint after t_g and at the end of training.
"""
import argparse
import os

import numpy as np
import torch

from analysis.analyze_mlp import load_exp
from analysis.analyze_revision import mlp_criteria
from grokking.common_mlp import grokking_onset, model_from_weights, run_data

LR = 1e-2
EXPS = [("loss_ce_lr0.01", 1e-2)] + [(f"ce_lr0.01_wd{wd:g}", wd) for wd in (3e-2, 1e-1, 3e-1, 1.0)]


def ms(vals, nd=1):
    v = [x for x in vals if x is not None]
    if not v:
        return "--"
    if len(v) == 1:
        return f"{v[0]:.{nd}f} (n=1)"
    return f"{np.mean(v):.{nd}f} +- {np.std(v, ddof=1):.{nd}f} (n={len(v)})"


def counts(r, step):
    X, y, tr, te = run_data(r)
    w = r["ckpts"][step]
    m = model_from_weights(w["W1"], w["W2"], r["config"]["act"])
    d, acc, _ = mlp_criteria(m, w["W1"].float(), X[te], y[te], centered=True)
    return d["d_out10"], d["d_acc"], acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(args.runs, "analysis_ce_wd")
    os.makedirs(out, exist_ok=True)
    torch.set_grad_enabled(False)

    lines = ["Cross-entropy MLP, lr 1e-2: weight decay vs crystallization (W1, test inputs, centered logits)",
             "d_out: fewest top directions keeping the outputs within 10%; d_0.99: keeping 99% of test accuracy", ""]
    for exp, wd in EXPS:
        runs = load_exp(args.runs, exp)
        lines.append(f"[{exp}]  wd {wd:g}  per-step decay lr*wd = {LR * wd:.0e}")
        if not runs:
            lines.append("  no runs found\n")
            continue
        tg, acc_end, er_end = [], [], []
        dout_aft, dacc_aft, dout_end, dacc_end = [], [], [], []
        for r in runs:
            h = r["hist"]
            t = grokking_onset(h["steps"], h["test_acc"])
            end = r["final_step"]
            de, da, a = counts(r, end)
            acc_end.append(a)
            er_end.append(float(h["W1_erank"][-1]))
            if t is None:
                continue
            tg.append(t)
            dout_end.append(de)
            dacc_end.append(da)
            after = [s for s in r["ckpts"] if s >= t and s != end]
            if after:
                o, c, _ = counts(r, min(after))
                dout_aft.append(o)
                dacc_aft.append(c)
        lines += [
            f"  grokked              {len(tg)}/{len(runs)}",
            f"  t_g                  {ms(tg, 0)}",
            f"  final test acc (%)   {ms(acc_end, 1)}   (all seeds)",
            f"  final ER(W1)         {ms(er_end, 1)}   (all seeds)",
            f"  d_out, first ckpt after t_g   {ms(dout_aft)}",
            f"  d_0.99, first ckpt after t_g  {ms(dacc_aft)}",
            f"  d_out, end           {ms(dout_end)}",
            f"  d_0.99, end          {ms(dacc_end)}",
            ""]
    text = "\n".join(lines)
    with open(os.path.join(out, "summary.txt"), "w") as f:
        f.write(text + "\n")
    print(text)
    print(f"wrote {out}/summary.txt")


if __name__ == "__main__":
    main()