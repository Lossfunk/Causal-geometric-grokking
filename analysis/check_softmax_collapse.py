"""
check_softmax_collapse.py -- did the fp32 softmax-collapse bug (Barron et al., 2026) act in our
cross-entropy runs?

  python -m analysis.check_softmax_collapse --runs results --holdout results_holdout

For every saved checkpoint of the cross-entropy runs (transformer at weight decay 1.0, and the
cross-entropy MLPs), on the training inputs: the median and largest margin (correct logit minus
the largest other logit) and the share of examples whose fp32 gradient of the cross-entropy with
respect to the correct logit is exactly zero (the collapse condition, roughly margin > 16.6).
From the logged histories: the largest jump of the training loss above its running minimum after
memorization, the signature of slingshot spikes. Writes <runs>/analysis_softmax_collapse/summary.txt.
"""
import argparse
import glob
import os

import numpy as np
import torch

from analysis.analyze_mlp import load_exp
from grokking.common_mlp import model_from_weights, run_data
from train.run_transformer import build_dataset, build_model

CE_MLP_EXPS = ["loss_ce", "loss_ce_lr0.003", "loss_ce_lr0.01",
               "ce_lr0.01_wd0.03", "ce_lr0.01_wd0.1", "ce_lr0.01_wd0.3", "ce_lr0.01_wd1"]


def collapse_stats(logits, y):
    logits = logits.float()
    zt = logits.gather(1, y[:, None]).squeeze(1)
    rest = logits.clone()
    rest.scatter_(1, y[:, None], -float("inf"))
    margin = zt - rest.max(1).values
    p_t = torch.softmax(logits, dim=1).gather(1, y[:, None]).squeeze(1)
    zero = float(((p_t - 1.0) == 0).float().mean())      # fp32 target gradient rounds to 0
    return float(margin.median()), float(margin.max()), zero


def spike(h):
    acc, loss, steps = (np.asarray(h[k], dtype=float) for k in ("train_acc", "train_loss", "steps"))
    if acc.max() > 1.5:                                   # stored in percent
        acc = acc / 100
    idx = np.where(acc >= 0.99)[0]
    if len(idx) == 0:
        return None
    i0 = idx[0]
    run_min = np.minimum.accumulate(loss[i0:])
    ratio = loss[i0:] / np.maximum(run_min, 1e-30)
    j = int(np.argmax(ratio))
    return float(ratio[j]), int(steps[i0 + j]), float(loss.min())


def worst(rows):
    """rows: (step, median margin, max margin, zero share); the checkpoint with the largest share."""
    return max(rows, key=lambda r: (r[3], r[2]))


def fmt_run(label, rows, sp):
    s, med, mx, z = worst(rows)
    fs, fmed, fmx, fz = rows[-1]
    sp_txt = "--" if sp is None else f"x{sp[0]:.1f} at step {sp[1]} (min loss {sp[2]:.1e})"
    return (f"  {label:<34} end: median margin {fmed:6.2f}, max {fmx:6.2f}, collapsed {100 * fz:5.1f}% | "
            f"worst ckpt {s}: collapsed {100 * z:5.1f}% (max margin {mx:.2f}) | loss jump {sp_txt}")


def transformer_section(lines, root, tag):
    paths = sorted(glob.glob(os.path.join(root, "transformer", "wd1", "seed*.pt")))
    lines.append(f"[transformer, weight decay 1.0, {tag}: {len(paths)} runs]")
    agg = []
    for path in paths:
        run = torch.load(path, map_location="cpu", weights_only=False)
        model = build_model(run["seed"], "cpu")
        model.eval()
        data, labels, _, _ = build_dataset(run["seed"], run["config"]["frac"], "cpu")
        X, y = data[run["train_idx"]], labels[run["train_idx"]]
        rows = []
        with torch.no_grad():
            for s in sorted(run["ckpts"]):
                model.load_state_dict(run["ckpts"][s])
                rows.append((s,) + collapse_stats(model(X)[:, -1, :], y))
        sp = spike(run["hist"])
        lines.append(fmt_run(f"seed {run['seed']}", rows, sp))
        agg.append((rows, sp))
    return agg


def mlp_section(lines, root):
    agg = []
    for exp in CE_MLP_EXPS:
        runs = load_exp(root, exp)
        if not runs:
            continue
        lines.append(f"[MLP {exp}: {len(runs)} runs]")
        for r in runs:
            X, y, tr, te = run_data(r)
            rows = []
            with torch.no_grad():
                for s in sorted(r["ckpts"]):
                    w = r["ckpts"][s]
                    m = model_from_weights(w["W1"], w["W2"], r["config"]["act"])
                    W1 = w["W1"].float()
                    lg = m.readout(m.act((W1 @ X[tr].T).T / np.sqrt(X.shape[1])))
                    rows.append((s,) + collapse_stats(lg, y[tr]))
            sp = spike(r["hist"])
            lines.append(fmt_run(f"seed {r['seed']}", rows, sp))
            agg.append((rows, sp))
    return agg


def verdict(lines, name, agg):
    if not agg:
        return
    any_collapse = sum(worst(rows)[3] > 0 for rows, _ in agg)
    big_jump = sum(sp is not None and sp[0] > 10 for _, sp in agg)
    max_margin = max(worst(rows)[2] for rows, _ in agg)
    lines.append(f"  -> {name}: runs with any collapsed examples at a saved checkpoint: {any_collapse}/{len(agg)};"
                 f" runs with a loss jump > 10x after memorization: {big_jump}/{len(agg)};"
                 f" largest margin seen: {max_margin:.2f} (fp32 cliff ~16.6)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results")
    ap.add_argument("--holdout", default="results_holdout")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(args.runs, "analysis_softmax_collapse")
    os.makedirs(out, exist_ok=True)
    torch.set_grad_enabled(False)
    lines = ["Softmax collapse check (fp32): share of training examples whose cross-entropy gradient",
             "for the correct logit is exactly zero; margins = correct logit - largest other logit.", ""]
    tf = transformer_section(lines, args.runs, "seeds 0-14")
    if os.path.isdir(args.holdout):
        tf += transformer_section(lines, args.holdout, "held-out seeds")
    mlp = mlp_section(lines, args.runs)
    lines.append("")
    verdict(lines, "transformer", tf)
    verdict(lines, "cross-entropy MLP", mlp)
    text = "\n".join(lines)
    with open(os.path.join(out, "summary.txt"), "w") as f:
        f.write(text + "\n")
    print(text)
    print(f"\nwrote {out}/summary.txt")


if __name__ == "__main__":
    main()