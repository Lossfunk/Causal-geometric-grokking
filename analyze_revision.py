"""
analyze_revision.py -- analyses requested in the third review, from saved runs.

  python analyze_revision.py --runs results --holdout results_holdout --dense results_dense
  python analyze_revision.py --runs results --sections A,B,C      # a subset

Writes <out>/summary.txt, summary.json and fig_*.png (default out: <runs>/analysis_revision).
Sections run only if their runs exist:

  A  Transformer: is compression locked to generalization or to fitting? (histories only;
     main seeds 0-14 + held-out seeds 15-29). Correlations of t_s with t_g and with the
     training-fit time, onset of the effective-rank drop, share of the drop after t_g with the
     initialization decay excluded, and the effective rank of the runs that never generalize.
  B  MLP: effective-rank trajectories of the random-label networks (fit, never generalize)
     next to the grokked ones.
  C  MLP: baselines for the margin-free count (step 0, random labels, train vs test inputs),
     two intermediate criteria (centered outputs, per-input margins), and the cross-entropy MLPs.
  D  Transformer, dense checkpoints (results_dense): margin-free count at every checkpoint
     around each seed's t_g, its lead or lag relative to t_g, the restricted / excluded loss of
     Nanda et al., and the same counts for the runs that never generalize.
  E  MLP: linear CKA between seeds and with controls; transplant controls (random orthogonal
     map, a grokked model of a different task, a randomly initialized model).
  F  MLP: gradient and Adam state during Stage 4 (radial component, Euler check, per-coordinate
     sqrt(v) against eps, how sign-like the update is, top-65 vs remaining directions).
  G  MLP: factorized runs logged every 50 steps (fine_* in run_mlp.py).
  H  Transformer: rank caps on W_E / W_in against a factorized full-rank control
     (run_transformer.py --factor).
  I  Synthetic check of the steepest-drop measure (no data needed).
"""
import argparse
import glob
import os
import re
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

from analyze_mlp import Report, clean, fmt, load_exp, post_act, savefig
from common_mlp import (P, accuracy, compression_onset, effective_rank, final_model, grokking_onset,
                        load_run, model_from_weights, run_data)

START_TF, START_MLP = 4000, 1000


# ================================================================ shared helpers
def smoothed_rate(steps, v, smooth=5):
    """Centered moving average over `smooth` logged points, then central differences: the
    same operations as steepest_drop() in analyze_mlp.py / analyze_transformer.py."""
    st, v = np.asarray(steps, float), np.asarray(v, float)
    return np.gradient(np.convolve(v, np.ones(smooth) / smooth, mode="same"), st)


def valid_mask(steps, start):
    st = np.asarray(steps)
    return (st >= start) & (st <= st[-1] - 4 * (st[1] - st[0]))


def steepest_drop(steps, v, start):
    d, ok, st = smoothed_rate(steps, v), valid_mask(steps, start), np.asarray(steps)
    return int(st[ok][np.argmin(d[ok])])


def drop_onset(steps, v, start, frac=0.2):
    """Start of the steepest drop: walking back from the steepest point, the first logged step
    of the contiguous stretch in which the smoothed rate is at least `frac` of the peak rate."""
    d, ok, st = smoothed_rate(steps, v), valid_mask(steps, start), np.asarray(steps)
    idx = np.where(ok)[0]
    i = idx[np.argmin(d[ok])]
    j = i
    while j - 1 >= idx[0] and d[j - 1] <= frac * d[i]:
        j -= 1
    return int(st[j]), float(d[i])


def first_step(steps, vals, thr):
    for s, a in zip(steps, vals):
        if a >= thr:
            return int(s)
    return None


def share_after(steps, er, t, base_from=0):
    """Share of the decrease from max(ER over [base_from, t]) to the final ER that happens after t."""
    st, er = np.asarray(steps), np.asarray(er, float)
    win = (st >= base_from) & (st <= t)
    if not win.any():
        return None
    peak, at, end = er[win].max(), er[st <= t][-1], er[-1]
    return float((at - end) / (peak - end)) if peak > end else None


def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return "n/a"
    r, p = stats.pearsonr(x, y)
    rho, q = stats.spearmanr(x, y)
    return f"Pearson r={r:.2f} (p={p:.1e}), Spearman rho={rho:.2f} (p={q:.1e})"


def partial_corr(x, y, z):
    """Correlation of x and y after removing the linear effect of z from both."""
    x, y, z = (np.asarray(a, float) for a in (x, y, z))
    if len(x) < 4 or z.std() == 0:
        return float("nan")
    rx = x - np.polyval(np.polyfit(z, x, 1), z)
    ry = y - np.polyval(np.polyfit(z, y, 1), z)
    return float(stats.pearsonr(rx, ry)[0])


def load_tf(root, sub="transformer"):
    """{(label, wd): [runs]} for <root>/<sub>/wd*/seed*.pt, read without transformer_lens."""
    out = {}
    for d in sorted(glob.glob(os.path.join(root, sub, "*"))):
        paths = sorted(glob.glob(os.path.join(d, "**", "seed*.pt"), recursive=True),
                       key=lambda p: int(re.findall(r"seed(\d+)", p)[-1]))
        if paths:
            out[os.path.relpath(d, os.path.join(root, sub))] = \
                [torch.load(p, map_location="cpu", weights_only=False) for p in paths]
    return out


def tf_grokked(run):
    return float(run["hist"]["test_acc"][-1]) >= 0.95


def tg_of(h):
    return first_step(h["steps"], h["test_acc"], 0.95)


# ================================================================ A. transformer: generalization or fitting?
def section_tf_timing(R, root, holdout, out):
    main = load_tf(root).get("wd1", [])
    held = load_tf(holdout).get("wd1", []) if holdout else []
    if not main and not held:
        return
    R.h("A. Transformer: is the effective-rank drop locked to generalization or to fitting?")
    seen, runs = set(), []
    for src, rs in (("main", main), ("held-out", held)):
        for r in rs:
            if r["seed"] not in seen:
                seen.add(r["seed"])
                runs.append((src, r))
    rows, neg = [], []
    for src, r in runs:
        h = r["hist"]
        st, er = h["steps"], h["W_in_erank"]
        on, rate = drop_onset(st, er, START_TF)
        row = dict(src=src, seed=r["seed"], tg=tg_of(h), ts=steepest_drop(st, er, START_TF), ton=on,
                   rate=rate, tfit99=first_step(st, h["train_acc"], 0.99),
                   tfit100=first_step(st, h["train_acc"], 1.0),
                   t10=first_step(st, h["test_acc"], 0.10), t50=first_step(st, h["test_acc"], 0.50),
                   acc_on=100 * float(np.interp(on, st, h["test_acc"])),
                   er4k=float(np.interp(START_TF, st, er)), er_end=float(er[-1]), er0=float(er[0]))
        if tf_grokked(r):
            row["share_raw"] = share_after(st, er, row["tg"])
            row["share_post"] = share_after(st, er, row["tg"], base_from=START_TF)
            rows.append(row)
        else:
            row["test_end"] = float(h["test_acc"][-1])
            neg.append((f"wd1 seed {r['seed']} ({src})", row))
    for wd, rs in load_tf(root).items():
        if wd == "wd1":
            continue
        for r in rs:
            if tf_grokked(r):
                continue
            h = r["hist"]
            st, er = h["steps"], h["W_in_erank"]
            on, rate = drop_onset(st, er, START_TF)
            neg.append((f"{wd} seed {r['seed']}", dict(ts=steepest_drop(st, er, START_TF), ton=on, rate=rate,
                        tfit99=first_step(st, h["train_acc"], 0.99), er4k=float(np.interp(START_TF, st, er)),
                        er_end=float(er[-1]), er0=float(er[0]), test_end=float(h["test_acc"][-1]))))
    g = lambda k: [w[k] for w in rows]
    R.p(f"  {len(rows)} grokked seeds (main + held-out); training-fit time = first logged step with")
    R.p(f"  train accuracy >= 99% (t_fit99) or 100% (t_fit100); t_s = steepest smoothed ER(W_in) decrease")
    R.p(f"  after step {START_TF:,}; t_on = start of that drop (rate >= 20% of its peak, walking back from t_s)")
    R.p(f"  t_10 / t_50 = first logged step with test accuracy >= 10% / 50%; acc(t_on) = test accuracy at t_on")
    R.p(f"  {'seed':>5} {'src':>9} {'t_fit99':>8} {'t_10':>7} {'t_50':>7} {'t_g':>7} {'t_on':>7} {'t_s':>7} "
        f"{'acc(t_on)':>10} {'t_on-t_g':>9} {'t_s-t_g':>8} {'share>t_g':>10}")
    for w in sorted(rows, key=lambda w: w["seed"]):
        sp = "n/a" if w["share_post"] is None else f"{w['share_post']:.2f}"
        R.p(f"  {w['seed']:>5} {w['src']:>9} {str(w['tfit99']):>8} {str(w['t10']):>7} {str(w['t50']):>7} "
            f"{w['tg']:>7} {w['ton']:>7} {w['ts']:>7} {w['acc_on']:>9.1f}% {w['ton'] - w['tg']:>+9} "
            f"{w['ts'] - w['tg']:>+8} {sp:>10}")
    tg, ts, ton, tf = np.array(g("tg")), np.array(g("ts")), np.array(g("ton")), g("tfit99")
    R.p(f"\n  t_g {fmt(tg, 0)};  t_fit99 {fmt(tf, 0)};  t_fit100 {fmt(g('tfit100'), 0)}")
    R.p(f"  t_s vs t_g:      {corr(ts, tg)}")
    if all(x is not None for x in tf):
        tf = np.array(tf, float)
        R.p(f"  t_s vs t_fit99:  {corr(ts, tf)}")
        R.p(f"  t_g vs t_fit99:  {corr(tg, tf)}")
        R.p(f"  partial r(t_s, t_g | t_fit99) = {partial_corr(ts, tg, tf):.2f};  "
            f"partial r(t_s, t_fit99 | t_g) = {partial_corr(ts, tf, tg):.2f}")
        R.p(f"  spread: s.d.(t_s - t_g) = {np.std(ts - tg, ddof=1):.0f}, s.d.(t_s - t_fit99) = "
            f"{np.std(ts - tf, ddof=1):.0f} steps")
    sl = stats.linregress(tg, ts)
    R.p(f"  regression t_s = {sl.intercept:.0f} + {sl.slope:.3f} t_g  (slope 95% CI "
        f"[{sl.slope - 1.96 * sl.stderr:.3f}, {sl.slope + 1.96 * sl.stderr:.3f}])")
    R.p(f"  drop onset t_on - t_g: {fmt(ton - tg, 0)};  t_on >= t_g in {int((ton >= tg).sum())}/{len(ton)}"
        f";  median {np.median(ton - tg):.0f}")
    for k in ("t10", "t50"):
        x = [w["ton"] - w[k] for w in rows if w[k] is not None]
        R.p(f"  t_on - {k}: {fmt(x, 0)};  t_on >= {k} in {sum(v >= 0 for v in x)}/{len(x)};  median "
            f"{np.median(x) if x else float('nan'):.0f}")
    R.p(f"  accuracy rise: t_50 - t_10 {fmt([w['t50'] - w['t10'] for w in rows if w['t10'] and w['t50']], 0)}, "
        f"t_g - t_10 {fmt([w['tg'] - w['t10'] for w in rows if w['t10']], 0)}")
    R.p(f"  test accuracy at the drop onset: {fmt(g('acc_on'), 1)} %;  share of the post-plateau drop before t_g: "
        f"{fmt([1 - w['share_post'] for w in rows if w['share_post'] is not None], 2)}")
    R.p(f"  share of the ER(W_in) decrease after t_g: including the initialization decay "
        f"{fmt(g('share_raw'), 2)};  from the plateau after step {START_TF:,} {fmt(g('share_post'), 2)}")
    R.p(f"  peak smoothed rate of ER(W_in) after step {START_TF:,} (ER units per 1,000 steps): grokked "
        f"{fmt([1000 * x for x in g('rate')], 2)}")
    R.p(f"  ER(W_in) from step {START_TF:,} to the end, relative change: grokked "
        f"{fmt([(w['er_end'] - w['er4k']) / w['er4k'] for w in rows], 3)}")
    if neg:
        R.p("\n  runs that never generalize:")
        R.p(f"  {'run':<26} {'test acc end':>12} {'t_fit99':>8} {'peak rate /1k':>14} {'ER 4k->end':>14}")
        for lab, w in neg:
            R.p(f"  {lab:<26} {100 * w['test_end']:>11.1f}% {str(w['tfit99']):>8} {1000 * w['rate']:>14.2f} "
                f"{(w['er_end'] - w['er4k']) / w['er4k']:>+14.3f}")
    R.data["A_tf_timing"] = dict(rows=rows, neg=neg)
    if all(w["tfit99"] is not None for w in rows):
        fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
        for ax, x, lab in ((axs[0], tg, "t_g (step of 95% test accuracy)"),
                           (axs[1], np.array(g("tfit99")), "t_fit (step of 99% train accuracy)")):
            c = ["C0" if w["src"] == "main" else "C1" for w in rows]
            ax.scatter(x, ts, c=c, s=18)
            lo, hi = min(x.min(), ts.min()), max(x.max(), ts.max())
            ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
            ax.set_xlabel(lab)
            ax.set_ylabel("t_s (steepest ER(W_in) decrease)")
        axs[0].set_title("blue: seeds 0-14, orange: held-out 15-29", fontsize=8)
        savefig(out, "fig_tf_ts_vs_tg_tfit.png")


TF_KEYS = {"W_E": "embed.W_E", "W_in": "blocks.0.mlp.W_in", "W_out": "blocks.0.mlp.W_out", "W_U": "unembed.W_U"}


def k99(W):
    s2 = torch.linalg.svdvals(W.float()) ** 2
    return int(torch.searchsorted(torch.cumsum(s2, 0) / s2.sum(), torch.tensor(0.99)).item()) + 1


def section_tf_matched(R, root, holdout):
    """Grokked seeds before their own onset vs runs that never generalize, at the same step."""
    groups = {}
    for src in (root, holdout):
        if not src:
            continue
        for wd, rs in load_tf(src).items():
            for r in rs:
                if r["seed"] in groups.get("_seen_" + wd, set()):
                    continue
                groups.setdefault("_seen_" + wd, set()).add(r["seed"])
                tg = tg_of(r["hist"]) if tf_grokked(r) else None
                for s, sd in r["ckpts"].items():
                    if s == r["final_step"] or s < START_TF:
                        continue
                    if tg is not None:
                        lab = "grokked, before own t_g" if s < tg else "grokked, after own t_g"
                    else:
                        lab = f"never generalize ({wd})"
                    h = r["hist"]
                    i = list(h["steps"]).index(s) if s in set(h["steps"].tolist()) else None
                    row = groups.setdefault((s, lab), {m: [] for m in TF_KEYS} | {"er": [], "acc": []})
                    for m, key in TF_KEYS.items():
                        W = sd[key][:-1] if m == "W_E" else sd[key]
                        row[m].append(k99(W))
                    if i is not None:
                        row["er"].append(float(h["W_in_erank"][i]))
                        row["acc"].append(100 * float(h["test_acc"][i]))
    rows = {k: v for k, v in groups.items() if not str(k).startswith("_seen_")}
    if not rows:
        return
    R.h("A2. Transformer: grokked runs before their onset vs runs that never generalize, at the same step")
    R.p("  k99 = singular directions carrying 99% of sum sigma^2 (weights at saved checkpoints); ER(W_in) from")
    R.p("  the history; 'never generalize' at wd 1 is the matched-weight-decay comparison")
    R.p(f"  {'step':>6} {'group':<32}" + "".join(f"{m:>20}" for m in TF_KEYS) + f"{'ER(W_in)':>20}{'test acc %':>20}")
    for (s, lab) in sorted(rows, key=lambda k: (k[0], k[1])):
        v = rows[(s, lab)]
        R.p(f"  {s:>6} {lab:<32}" + "".join(f"{fmt(v[m], 1):>20}" for m in TF_KEYS)
            + f"{fmt(v['er'], 1):>20}{fmt(v['acc'], 1):>20}")
    R.data["A2_tf_matched"] = {f"{s}|{lab}": v for (s, lab), v in rows.items()}


# ================================================================ B. MLP: random labels
def section_mlp_fit(R, root, out):
    main, rand = load_exp(root, "main"), load_exp(root, "random_labels")
    if not main:
        return
    R.h("B. MLP: effective rank of networks that fit without generalizing (random labels)")

    def row(r):
        h = r["hist"]
        st, er = h["steps"], h["W1_erank"]
        tfit = first_step(st, h["train_acc"], 0.99)
        d = dict(seed=r["seed"], train_end=float(h["train_acc"][-1]), test_end=float(h["test_acc"][-1]),
                 tfit99=tfit, tfit90=first_step(st, h["train_acc"], 0.90), tg=grokking_onset(st, h["test_acc"]),
                 tc=compression_onset(st, er), ts=steepest_drop(st, er, START_MLP),
                 er_max=float(np.max(er)), er_end=float(er[-1]))
        d["rel_drop"] = (d["er_max"] - d["er_end"]) / d["er_max"]
        ref = tfit if tfit is not None else d["tfit90"]
        d["share_after_fit"] = share_after(st, er, ref) if ref is not None else None
        return d

    for lab, runs in (("grokked (main)", main), ("random labels", rand)):
        if not runs:
            continue
        rows = [row(r) for r in runs]
        g = lambda k: [w[k] for w in rows]
        R.p(f"\n  [{lab}]  n={len(rows)}")
        R.p(f"    final train acc {fmt([100 * x for x in g('train_end')], 1)} %;  final test acc "
            f"{fmt([100 * x for x in g('test_end')], 1)} %")
        R.p(f"    first step with train acc >= 90%: {fmt(g('tfit90'), 0)};  >= 99%: {fmt(g('tfit99'), 0)}")
        R.p(f"    t_g {fmt(g('tg'), 0)};  t_c {fmt(g('tc'), 0)};  steepest ER decrease t_s {fmt(g('ts'), 0)}")
        R.p(f"    ER(W1) max {fmt(g('er_max'), 1)} -> end {fmt(g('er_end'), 1)};  relative drop "
            f"{fmt([100 * x for x in g('rel_drop')], 1)} %")
        R.p(f"    share of the ER decrease after the fit time (99%, or 90% if never reached): "
            f"{fmt(g('share_after_fit'), 2)}")
        R.data.setdefault("B_mlp_fit", {})[lab] = rows
        if lab.startswith("grokked"):
            tf = g("tfit99")
            if all(x is not None for x in tf):
                R.p(f"    t_s vs t_g: {corr(g('ts'), g('tg'))}")
                R.p(f"    t_s vs t_fit99: {corr(g('ts'), tf)}  (t_g and t_fit barely vary across MLP seeds,")
                R.p(f"    so the MLP cannot separate generalization from fitting)")
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    for runs, col, lab in ((main, "C0", "grokked"), (rand, "C3", "random labels")):
        for i, r in enumerate(runs):
            h = r["hist"]
            axs[0].plot(h["steps"], h["W1_erank"], color=col, lw=0.7, alpha=0.6, label=lab if i == 0 else None)
            axs[1].plot(h["steps"], h["train_acc"], color=col, lw=0.7, alpha=0.6, label=lab if i == 0 else None)
    axs[0].set_ylabel("effective rank of W1")
    axs[1].set_ylabel("train accuracy")
    for ax in axs:
        ax.set_xlabel("step")
        ax.set_xscale("symlog", linthresh=1000)
        ax.legend(fontsize=7)
    savefig(out, "fig_mlp_er_random_labels.png")


# ================================================================ C. MLP margin-free baselines
def mlp_criteria(model, W1, X, y, centered=False):
    """Smallest top-k truncation of W1 meeting each criterion (k = 1..rank, early stop).
    d_acc: 99% of the accuracy; d_out10 / d_out05: outputs within 10% / 5% (relative Frobenius
    norm; centered per input if `centered`, as for logits); d_cen10: per-input centered outputs
    within 10%; d_marg10: the vector of per-input margins within 10%."""
    U, S, Vh = torch.linalg.svd(W1, full_matrices=False)
    fwd = lambda W: model.readout(model.act((W @ X.T).T / np.sqrt(X.shape[1])))
    cen = lambda f: f - f.mean(1, keepdim=True)

    def margins(f):
        fc = f.gather(1, y[:, None]).squeeze(1)
        rest = f.clone()
        rest.scatter_(1, y[:, None], -float("inf"))
        return fc - rest.max(1).values

    with torch.no_grad():
        f0 = fwd(W1)
        g0 = cen(f0) if centered else f0
        c0, m0 = cen(f0), margins(f0)
        acc0 = accuracy(f0, y)
        d = dict(d_acc=None, d_out10=None, d_out05=None, d_cen10=None, d_marg10=None)
        for k in range(1, len(S) + 1):
            fk = fwd((U[:, :k] * S[:k]) @ Vh[:k])
            gk = cen(fk) if centered else fk
            rel = float((gk - g0).norm() / g0.norm())
            for key, ok in (("d_acc", accuracy(fk, y) >= 0.99 * acc0), ("d_out10", rel <= 0.10),
                            ("d_out05", rel <= 0.05),
                            ("d_cen10", float((cen(fk) - c0).norm() / c0.norm()) <= 0.10),
                            ("d_marg10", float((margins(fk) - m0).norm() / m0.norm()) <= 0.10)):
                if d[key] is None and ok:
                    d[key] = k
            if all(v is not None for v in d.values()):
                break
    return d, 100 * acc0, float(m0.median())


def section_mlp_baselines(R, root):
    main = load_exp(root, "main")
    if not main:
        return
    R.h("C. MLP: baselines and intermediate criteria for the margin-free count")
    R.p("  d_acc: 99% of accuracy; out10/out05: outputs within 10%/5%; cen10: per-input centered outputs")
    R.p("  within 10%; marg10: per-input margins within 10% (relative norms). CE models: out* use centered logits.")
    cols = ["d_acc", "d_out10", "d_out05", "d_cen10", "d_marg10"]

    def block(label, items, centered=False):
        res = {c: [] for c in cols + ["acc", "margin"]}
        for r, s, split in items:
            X, y, tr, te = run_data(r)
            idx = te if split == "test" else tr
            w = r["ckpts"][s]
            m = model_from_weights(w["W1"], w["W2"], r["config"]["act"])
            d, a, mg = mlp_criteria(m, w["W1"].float(), X[idx], y[idx], centered)
            for c in cols:
                res[c].append(d[c])
            res["acc"].append(a)
            res["margin"].append(mg)
        if not res["acc"]:
            return
        R.p(f"  {label:<46} acc {fmt(res['acc'], 1):>20}  margin {fmt(res['margin'], 3):>22}")
        R.p("      " + "  ".join(f"{c[2:]} {fmt(res[c], 1)}" for c in cols))
        R.data.setdefault("C_baselines", {})[label] = res

    last = lambda r: r["final_step"]
    block("main, step 0 (untrained), test inputs", [(r, 0, "test") for r in main if 0 in r["ckpts"]])
    for s in (11000, 20000):
        block(f"main, step {s:,}, test inputs", [(r, s, "test") for r in main if s in r["ckpts"]])
    block("main, end, test inputs", [(r, last(r), "test") for r in main])
    block("main, end, train inputs", [(r, last(r), "train") for r in main])
    rand = load_exp(root, "random_labels")
    block("random labels, step 0, train inputs", [(r, 0, "train") for r in rand if 0 in r["ckpts"]])
    block("random labels, end, train inputs", [(r, last(r), "train") for r in rand])
    block("random labels, end, test inputs", [(r, last(r), "test") for r in rand])
    for exp in ("mul_main", "p113_main", "loss_ce_lr0.003", "loss_ce_lr0.01", "loss_ce"):
        runs = [r for r in load_exp(root, exp) if grokking_onset(r["hist"]["steps"], r["hist"]["test_acc"])]
        if not runs:
            continue
        ce = runs[0]["config"]["loss"] == "ce"
        tag = "(CE)" if ce else "(MSE, pre-registered task)"
        before, after = [], []
        for r in runs:
            tg = grokking_onset(r["hist"]["steps"], r["hist"]["test_acc"])
            b = [s for s in r["ckpts"] if s < tg]
            a = [s for s in r["ckpts"] if s >= tg and s != r["final_step"]]
            if b:
                before.append((r, max(b), "test"))
            if a:
                after.append((r, min(a), "test"))
        mid = [(r, 20000, "test") for r in runs if 20000 in r["ckpts"]] if not ce else []
        for lab, items in (("last checkpoint before t_g", before), ("first checkpoint after t_g", after),
                           ("step 20,000", mid), ("end", [(r, last(r), "test") for r in runs])):
            if items:
                block(f"{exp} {tag}, {lab}", items, centered=ce)


# ================================================================ D. transformer dense checkpoints
def tf_counts(model, name, data, labels, set_weight, truncated, weight_matrices, as2d):
    """Smallest top-k truncation of one matrix meeting each criterion (every k, early stop)."""
    with torch.no_grad():
        base = model(data)[:, -1, :]
    base_c = base - base.mean(-1, keepdim=True)
    acc0 = float((base.argmax(-1) == labels).float().mean())
    rest = base.clone()
    rest.scatter_(1, labels[:, None], -float("inf"))
    margin = float((base.gather(1, labels[:, None]).squeeze(1) - rest.max(-1).values).median())
    W0 = weight_matrices(model)[name].detach().clone()
    U, S, Vh = torch.linalg.svd(as2d(W0).float(), full_matrices=False)
    d = dict(d_acc=None, d_out10=None, d_out05=None)
    for k in range(1, len(S) + 1):
        set_weight(model, name, ((U[:, :k] * S[:k]) @ Vh[:k]).reshape(W0.shape))
        with torch.no_grad():
            lg = model(data)[:, -1, :]
        rel = float(((lg - lg.mean(-1, keepdim=True)) - base_c).norm() / base_c.norm())
        for key, ok in (("d_acc", float((lg.argmax(-1) == labels).float().mean()) >= 0.99 * acc0),
                        ("d_out10", rel <= 0.10), ("d_out05", rel <= 0.05)):
            if d[key] is None and ok:
                d[key] = k
        if all(v is not None for v in d.values()):
            break
    set_weight(model, name, W0)
    return d, 100 * acc0, margin


def key_freqs_tf(W_E, p=P, K=5):
    pw = (np.abs(np.fft.fft(W_E.detach().cpu().numpy()[:p], axis=0)) ** 2).sum(1)
    fs = np.arange(1, (p - 1) // 2 + 1)
    return [int(f) for f in fs[np.argsort(pw[fs] + pw[p - fs])[::-1][:K]]]


def restricted_excluded_loss(logits, labels, freqs, p=P):
    """Nanda et al.'s restricted / excluded loss, per input in output space: keep only (restricted)
    or remove (excluded) the components of the centered digit logits along cos / sin of the key
    frequencies of the output class c."""
    z = logits[:, :p]
    c = torch.arange(p, dtype=torch.float32, device=z.device)
    B = torch.stack([f(2 * np.pi * w * c / p) for w in freqs for f in (torch.cos, torch.sin)], 1)
    Q, _ = torch.linalg.qr(B)
    mu = z.mean(1, keepdim=True)
    proj = (z - mu) @ Q @ Q.T
    return float(F.cross_entropy(mu + proj, labels)), float(F.cross_entropy(z - proj, labels))


def section_tf_dense(R, dense, device, out, window=3000):
    if not dense or not glob.glob(os.path.join(dense, "transformer", "wd*", "seed*.pt")):
        return
    from analyze_transformer import load_runs, restore, set_weight
    from run_transformer import as2d, weight_matrices
    trunc = None
    R.h("D. Transformer, dense checkpoints: margin-free count around each seed's t_g")
    mats = ("W_E", "W_in", "W_out", "W_U")
    runs = load_runs(dense, 1.0)
    grk = [r for r in runs if tf_grokked(r)]
    per = []
    for r in grk:
        tg = tg_of(r["hist"])
        steps = sorted(s for s in r["ckpts"] if tg - window <= s <= tg + window) + [r["final_step"]]
        m0, _, _, _, _ = restore(r, device)
        freqs = key_freqs_tf(weight_matrices(m0)["W_E"])
        rec = dict(seed=r["seed"], tg=tg, steps=steps, freqs=freqs, acc=[], margin=[], rl=[], el=[],
                   **{f"{m}|{c}": [] for m in mats for c in ("d_acc", "d_out10", "d_out05")})
        for s in steps:
            model, data, labels, tr, te = restore(r, device, s)
            with torch.no_grad():
                rl, el = restricted_excluded_loss(model(data[tr])[:, -1, :], labels[tr], freqs)
            rec["rl"].append(rl)
            rec["el"].append(el)
            for i, m in enumerate(mats):
                d, a, mg = tf_counts(model, m, data[te], labels[te], set_weight, trunc, weight_matrices, as2d)
                if i == 0:
                    rec["acc"].append(a)
                    rec["margin"].append(mg)
                for c in d:
                    rec[f"{m}|{c}"].append(d[c])
        per.append(rec)
        print(f"  dense seed {r['seed']} done", flush=True)
    if not per:
        return
    rel = sorted({s - p["tg"] for p in per for s in p["steps"][:-1]})
    grid = [x for x in rel if x % 500 == 0]
    R.p(f"  {len(per)} grokked seeds; checkpoints every 250 steps within +-{window:,} of t_g; counts on test inputs")
    R.p(f"  mean logits-10% count aligned at t_g (step - t_g), and test accuracy:")
    R.p("  " + f"{'step-t_g':>9}" + "".join(f"{m:>18}" for m in mats) + f"{'test acc %':>18}"
        + f"{'restricted L':>14}{'excluded L':>14}")
    for x in grid:
        vals = {m: [] for m in mats}
        accs, rls, els = [], [], []
        for p in per:
            if p["tg"] + x in p["steps"]:
                i = p["steps"].index(p["tg"] + x)
                for m in mats:
                    vals[m].append(p[f"{m}|d_out10"][i])
                accs.append(p["acc"][i])
                rls.append(p["rl"][i])
                els.append(p["el"][i])
        R.p("  " + f"{x:>+9}" + "".join(f"{fmt(vals[m], 1):>18}" for m in mats) + f"{fmt(accs, 1):>18}"
            + f"{np.mean(rls):>14.3f}{np.mean(els):>14.3f}")
    R.p("\n  lead/lag: t_f = first checkpoint at which the count falls below the midpoint between its value")
    R.p("  at the start of the window and its final value; t_r = same rule for the restricted loss;")
    R.p("  t_x = first checkpoint at which the excluded loss exceeds the midpoint of its rise.")
    lags = {m: [] for m in mats}
    lr_, lx = [], []

    def cross(steps, v, below=True):
        v = np.asarray(v, float)
        mid = 0.5 * (v[0] + v[-1])
        for s, a in zip(steps, v):
            if (a <= mid) if below else (a >= mid):
                return s
        return None

    for p in per:
        st = p["steps"]
        for m in mats:
            t = cross(st, p[f"{m}|d_out10"])
            if t is not None:
                lags[m].append(t - p["tg"])
        t = cross(st, p["rl"])
        if t is not None:
            lr_.append(t - p["tg"])
        el = np.asarray(p["el"], float)
        mid = 0.5 * (el[0] + el[:-1].max())
        tx = next((s for s, a in zip(st, el) if a >= mid), None)
        if tx is not None:
            lx.append(tx - p["tg"])
    for m in mats:
        x = clean(lags[m])
        R.p(f"  {m:<6} t_f - t_g {fmt(x, 0)};  before t_g in {int((x < 0).sum())}/{len(x)};  median "
            f"{np.median(x) if len(x) else float('nan'):.0f}")
    R.p(f"  restricted loss t_r - t_g {fmt(lr_, 0)};  excluded loss t_x - t_g {fmt(lx, 0)}")
    R.data["D_tf_dense"] = per
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    for m in mats:
        mu = [np.mean([p[f"{m}|d_out10"][p["steps"].index(p["tg"] + x)] for p in per if p["tg"] + x in p["steps"]])
              for x in rel]
        axs[0].plot(rel, mu, label=m)
    axs[1].plot(rel, [np.mean([p["acc"][p["steps"].index(p["tg"] + x)] for p in per if p["tg"] + x in p["steps"]])
                      for x in rel], color="k")
    axs[0].set_ylabel("directions keeping the logits within 10%")
    axs[1].set_ylabel("test accuracy %")
    for ax in axs:
        ax.axvline(0, color="grey", ls="--", lw=0.8)
        ax.set_xlabel("step - t_g")
    axs[0].legend(fontsize=7)
    savefig(out, "fig_tf_dense_margin_free.png")
    neg = [r for r in runs if not tf_grokked(r)]
    for wd in (0.1, 0.3):
        neg += [r for r in load_runs(dense, wd) if not tf_grokked(r)]
    if neg:
        R.p("\n  runs that never generalize: logits-10% count at fixed steps")
        R.p("  " + f"{'run':<22}{'step':>7}" + "".join(f"{m:>8}" for m in mats) + f"{'test acc %':>12}")
        for r in neg:
            for s in [s for s in (5000, 10000, 15000, 20000) if s in r["ckpts"]] + [r["final_step"]]:
                model, data, labels, tr, te = restore(r, device, s)
                ds, a = [], None
                for m in mats:
                    d, a0, _ = tf_counts(model, m, data[te], labels[te], set_weight, trunc, weight_matrices, as2d)
                    ds.append(d["d_out10"])
                    a = a0
                R.p("  " + f"{'wd' + str(r['config']['wd']) + ' seed ' + str(r['seed']):<22}{s:>7}"
                    + "".join(f"{x:>8}" for x in ds) + f"{a:>12.1f}")


# ================================================================ E. CKA and transplant controls
def linear_cka(X, Y):
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    return float((Y.T @ X).norm() ** 2 / ((X.T @ X).norm() * (Y.T @ Y).norm()))


def procrustes(ZA, ZB):
    U, _, Vt = torch.linalg.svd(ZA.T @ ZB)
    return U @ Vt


def section_cka(R, root):
    main = {r["seed"]: r for r in load_exp(root, "main")}
    if len(main) < 2:
        return
    R.h("E. MLP: linear CKA between hidden representations, and transplant controls")
    mul = {r["seed"]: r for r in load_exp(root, "mul_main")}
    rand = {r["seed"]: r for r in load_exp(root, "random_labels")}
    X, y_add, _, _ = run_data(next(iter(main.values())))
    seeds = sorted(main)
    Z = {s: post_act(final_model(main[s]), X) for s in seeds}
    init = lambda r: model_from_weights(r["ckpts"][0]["W1"], r["ckpts"][0]["W2"], r["config"]["act"])
    Z0 = {s: post_act(init(main[s]), X) for s in seeds if 0 in main[s]["ckpts"]}
    groups = {}
    for a, b in combinations(seeds, 2):
        groups.setdefault("grokked vs grokked (different seeds)", []).append(linear_cka(Z[a], Z[b]))
    for s in Z0:
        groups.setdefault("grokked vs its own initialization", []).append(linear_cka(Z[s], Z0[s]))
        o = seeds[(seeds.index(s) + 1) % len(seeds)]
        if o in Z0:
            groups.setdefault("grokked vs another seed's initialization", []).append(linear_cka(Z[s], Z0[o]))
    for a, b in combinations(sorted(Z0), 2):
        groups.setdefault("initialization vs initialization", []).append(linear_cka(Z0[a], Z0[b]))
    for s, r in mul.items():
        if s in Z:
            Zm = post_act(final_model(r), X)
            groups.setdefault("grokked a+b vs grokked ab, same seed", []).append(linear_cka(Z[s], Zm))
            o = seeds[(seeds.index(s) + 1) % len(seeds)]
            groups.setdefault("grokked a+b vs grokked ab, other seed", []).append(linear_cka(Z[o], Zm))
    for s, r in rand.items():
        if s in Z:
            groups.setdefault("grokked vs random-label model", []).append(linear_cka(Z[s], post_act(final_model(r), X)))
    R.p("  linear CKA of the post-activations on all p^2 inputs (1 = identical up to rotation and scale)")
    for k, v in groups.items():
        R.p(f"    {k:<46} {fmt(v, 3)}")
    R.data["E_cka"] = groups

    R.p("\n  transplants: decode model A's post-activations with model B's readout, after the orthogonal")
    R.p("  map fitted on B's training inputs; test accuracy on B's test inputs against the a+b labels")
    tg = {}
    rng = torch.Generator().manual_seed(0)

    def run_tp(mA, rB, R_map=None):
        mB = final_model(rB)
        _, _, tr, te = run_data(rB)
        ZA, ZB = post_act(mA, X), post_act(mB, X)
        Rm = procrustes(ZA[tr], ZB[tr]) if R_map is None else R_map
        with torch.no_grad():
            return accuracy(mB.readout(ZA[te] @ Rm), y_add[te])

    for a, b in zip(seeds[0::2], seeds[1::2]):
        for s, o in ((a, b), (b, a)):
            tg.setdefault("grokked -> grokked (other seed), fitted map", []).append(run_tp(final_model(main[s]), main[o]))
            Q, _ = torch.linalg.qr(torch.randn(128, 128, generator=rng))
            tg.setdefault("grokked -> grokked (other seed), random orthogonal map", []).append(
                run_tp(final_model(main[s]), main[o], Q))
            if 0 in main[s]["ckpts"]:
                tg.setdefault("random initialization -> grokked, fitted map", []).append(run_tp(init(main[s]), main[o]))
            if s in mul:
                tg.setdefault("grokked ab (other task) -> grokked a+b, fitted map", []).append(
                    run_tp(final_model(mul[s]), main[o]))
    for s, r in rand.items():
        o = seeds[(seeds.index(s) + 1) % len(seeds)] if s in seeds else None
        if o is not None:
            tg.setdefault("random-label model -> grokked, fitted map", []).append(run_tp(final_model(r), main[o]))
    for k, v in tg.items():
        R.p(f"    {k:<58} {fmt([100 * x for x in v], 2)} %")
    R.p(f"    chance {100 / P:.2f} %")
    R.data["E_transplant"] = tg


# ================================================================ F. Adam state in Stage 4
def section_adam_state(R, root, device, n_steps=1000, seeds=None):
    main = load_exp(root, "main")
    if not main:
        return
    R.h("F. MLP: gradient and Adam state during Stage 4 (continued for a short while from checkpoints)")
    R.p(f"  each checkpoint is continued for {n_steps} AdamW steps (lr 1e-3, wd 1e-2, eps 1e-8) to rebuild the")
    R.p("  optimizer state; then: |g| = full-batch gradient, cos(g, theta) = radial share of the gradient,")
    R.p("  Euler = 3 * (2/(p n)) sum (f - y) f (equals <g, theta> for this 3-homogeneous model),")
    R.p("  sqrt(v) quantiles against eps, sign-likeness = mean |m_hat / (sqrt(v_hat) + eps)| (1 = sign descent),")
    R.p("  radial Adam push <-u, theta> against the decay pull wd |theta|^2 (both per unit lr), and the share")
    R.p("  of |g_W1|^2 in the top-65 left singular directions of W1.")
    keys = ["gnorm", "gW1", "gW2", "cos", "radial", "euler", "v10", "v50", "v90", "below_eps", "below_100eps",
            "signlike", "adam_push", "sign_push", "decay_pull", "top65_share"]
    res = {}
    for r in main if seeds is None else [r for r in main if r["seed"] in seeds]:
        X, y, tr, _ = run_data(r, device)
        Xtr, Y = X[tr], F.one_hot(y[tr], P).float()
        for s in (20000, 50000, r["final_step"]):
            if s not in r["ckpts"]:
                continue
            w = r["ckpts"][s]
            m = model_from_weights(w["W1"], w["W2"], r["config"]["act"]).to(device)
            opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2, eps=1e-8)
            for _ in range(n_steps):
                opt.zero_grad()
                F.mse_loss(m(Xtr), Y).backward()
                opt.step()
            opt.zero_grad()
            out = m(Xtr)
            F.mse_loss(out, Y).backward()
            ps = [m.W1_full, m.W2]
            g = torch.cat([q.grad.flatten() for q in ps])
            th = torch.cat([q.detach().flatten() for q in ps])
            b1, b2 = opt.defaults["betas"]
            mh, vh = [], []
            for q in ps:
                stt = opt.state[q]
                t = float(stt["step"])
                mh.append((stt["exp_avg"] / (1 - b1 ** t)).flatten())
                vh.append((stt["exp_avg_sq"] / (1 - b2 ** t)).flatten())
            mh, vh = torch.cat(mh), torch.cat(vh)
            sv = vh.sqrt()
            u = mh / (sv + 1e-8)
            with torch.no_grad():
                euler = float(3 * 2 / Y.numel() * ((out - Y) * out).sum())
                U, _, _ = torch.linalg.svd(m.W1_full.detach(), full_matrices=False)
                gW1 = m.W1_full.grad
                top = float((U[:, :65].T @ gW1).norm() ** 2 / gW1.norm() ** 2)
            q = torch.quantile(sv.cpu(), torch.tensor([0.1, 0.5, 0.9]))
            row = dict(gnorm=float(g.norm()), gW1=float(m.W1_full.grad.norm()), gW2=float(m.W2.grad.norm()),
                       cos=float(g @ th / (g.norm() * th.norm())), radial=float(g @ th), euler=euler,
                       v10=float(q[0]), v50=float(q[1]), v90=float(q[2]),
                       below_eps=float((sv < 1e-8).float().mean()), below_100eps=float((sv < 1e-6).float().mean()),
                       signlike=float(u.abs().mean()), adam_push=float(-u @ th),
                       sign_push=float(-torch.sign(g) @ th), decay_pull=float(1e-2 * th @ th), top65_share=top)
            lab = "end" if s == r["final_step"] else f"{s:,}"
            for k in keys:
                res.setdefault(lab, {}).setdefault(k, []).append(row[k])
    for lab, d in res.items():
        R.p(f"\n  [step {lab}]  n={len(d['gnorm'])}")
        e = lambda v: f"{np.mean(v):.3e} ± {np.std(v, ddof=1):.1e} (n={len(v)})"
        R.p(f"    |g| {e(d['gnorm'])}  (W1 {e(d['gW1'])}, W2 {e(d['gW2'])})")
        R.p(f"    <g, theta> {e(d['radial'])};  Euler {e(d['euler'])};  cos(g, theta) {fmt(d['cos'], 4)}")
        R.p(f"    sqrt(v_hat) quantiles 10/50/90%: {np.mean(d['v10']):.2e} / {np.mean(d['v50']):.2e} / "
            f"{np.mean(d['v90']):.2e};  share below eps {fmt(d['below_eps'], 4)}, below 100 eps "
            f"{fmt(d['below_100eps'], 4)}")
        R.p(f"    sign-likeness {fmt(d['signlike'], 3)}")
        R.p(f"    radial push per unit lr: Adam {fmt(d['adam_push'], 1)}, pure sign descent {fmt(d['sign_push'], 1)};"
            f"  decay pull {fmt(d['decay_pull'], 1)}")
        R.p(f"    share of |g_W1|^2 in the top 65 directions {fmt(d['top65_share'], 3)}")
    R.data["F_adam_state"] = res


# ================================================================ G. fine-logged factorized runs
def section_fine(R, root, out):
    exps = ["fine_main", "fine_rank_128", "fine_rank_64", "fine_rank_32", "fine_mul_main", "fine_mul_rank_128",
            "fine_p113_main", "fine_p113_rank_128"]
    data = {e: {r["seed"]: r for r in load_exp(root, e)} for e in exps}
    if not any(data.values()):
        return
    R.h("G. MLP: factorized runs logged every 50 steps")
    tgs = {}
    for e, runs in data.items():
        if not runs:
            continue
        rows = []
        for s, r in sorted(runs.items()):
            h = r["hist"]
            st, er = np.asarray(h["steps"]), np.asarray(h["W1_erank"])
            tg = grokking_onset(st, h["test_acc"])
            pre = st <= (tg if tg is not None else st[-1])
            i = int(np.argmin(er[pre]))
            rows.append(dict(seed=s, tg=tg, er0=float(er[0]), dip=float(er[pre][i]), dip_step=int(st[pre][i]),
                             er_tg=float(er[st == tg][0]) if tg is not None else None, er_end=float(er[-1])))
        tgs[e] = {w["seed"]: w["tg"] for w in rows}
        g = lambda k: [w[k] for w in rows]
        R.p(f"  {e:<20} t_g {fmt(g('tg'), 0):>22}   ER: step 0 {fmt(g('er0'), 1)}, minimum before t_g "
            f"{fmt(g('dip'), 1)} at step {fmt(g('dip_step'), 0)}, at t_g {fmt(g('er_tg'), 1)}, end {fmt(g('er_end'), 1)}")
    for task, a, b in (("a+b mod 97", "fine_main", "fine_rank_128"), ("ab mod 97", "fine_mul_main", "fine_mul_rank_128"),
                       ("a+b mod 113", "fine_p113_main", "fine_p113_rank_128")):
        if tgs.get(a) and tgs.get(b):
            ratio = [tgs[a][s] / tgs[b][s] for s in tgs[b] if s in tgs[a] and tgs[a][s] and tgs[b][s]]
            R.p(f"  speed-up of the full-rank factorization, {task}: t_g ratio {fmt(ratio, 2)}")
    R.data["G_fine"] = tgs
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for e, col in (("fine_rank_128", "C0"), ("fine_rank_64", "C1"), ("fine_rank_32", "C2")):
        for i, r in enumerate(data.get(e, {}).values()):
            ax.plot(r["hist"]["steps"], r["hist"]["W1_erank"], color=col, lw=0.8, label=e if i == 0 else None)
    ax.set_xlabel("step")
    ax.set_ylabel("effective rank of W1 = AB")
    ax.legend(fontsize=7)
    savefig(out, "fig_fine_factorized_er.png")


# ================================================================ H. transformer rank caps
def section_tf_caps(R, root):
    caps = load_tf(root, "transformer_cap")
    if not caps:
        return
    R.h("H. Transformer: rank caps on W_E / W_in (W = A B) against a factorized full-rank control")
    base = load_tf(root).get("wd1", [])
    seeds = sorted({r["seed"] for rs in caps.values() for r in rs})
    rows = [("unfactorized (main runs)", "W_in", [r for r in base if r["seed"] in seeds])]
    for key in sorted(caps, key=lambda k: (k.split("_r")[0], -int(re.findall(r"_r(\d+)", k)[0]))):
        rows.append((key.split(os.sep)[0], key.split("_r")[0], caps[key]))
    R.p(f"  {'model':<26} {'grokked':>8} {'t_g':>22} {'final test %':>20} {'final ER(capped)':>20}")
    out = {}
    for lab, mat, rs in rows:
        if not rs:
            continue
        tg = [tg_of(r["hist"]) for r in rs if tf_grokked(r)]
        acc = [100 * float(r["hist"]["test_acc"][-1]) for r in rs]
        er = [float(r["hist"][f"{mat}_erank"][-1]) for r in rs]
        R.p(f"  {lab:<26} {len(tg):>4}/{len(rs):<3} {fmt(tg, 0):>22} {fmt(acc, 1):>20} {fmt(er, 1):>20}")
        out[lab] = dict(tg=tg, acc=acc, er=er)
    R.data["H_tf_caps"] = out


# ================================================================ I. synthetic check of the timing measure
def section_synthetic(R, out):
    R.h("I. Synthetic check of the steepest-drop measure")
    st = np.arange(0, 25001, 250)
    R.p("  logging every 250 steps, start 4,000; a 'step' drops at the first logged step >= t0,")
    R.p("  a 'ramp' falls linearly over 1,000 steps centered on t0")
    res = []
    for t0 in (8000, 12000, 12125, 17750):
        step = np.where(st < t0, 100.0, 60.0)
        ramp = np.interp(st, [t0 - 500, t0 + 500], [100, 60])
        a, b = steepest_drop(st, step, START_TF), steepest_drop(st, ramp, START_TF)
        res.append((t0, a, b))
        R.p(f"    t0={t0:>6}: step -> {a:>6} ({a - t0:+d}),  ramp -> {b:>6} ({b - t0:+d})")
    R.data["I_synthetic"] = res
    t0 = 12000
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, v, lab in ((axs[0], np.where(st < t0, 100.0, 60.0), "step"),
                       (axs[1], np.interp(st, [t0 - 500, t0 + 500], [100, 60]), "1,000-step ramp")):
        ax.plot(st, v, "k.-", ms=3, lw=0.8, label="signal")
        ax.plot(st, np.convolve(v, np.ones(5) / 5, mode="same"), "C0", lw=1, label="centered 5-point average")
        ax.axvline(t0, color="grey", ls="--", lw=0.8)
        ax.axvline(steepest_drop(st, v, START_TF), color="C3", lw=1, label="detected t_s")
        ax.set_xlim(t0 - 3000, t0 + 3000)
        ax.set_ylim(50, 110)
        ax.set_title(lab, fontsize=9)
        ax.set_xlabel("step")
    axs[0].legend(fontsize=7, loc="lower left")
    savefig(out, "fig_smoothing_check.png")


# ================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results")
    ap.add_argument("--holdout", default="results_holdout")
    ap.add_argument("--dense", default="results_dense")
    ap.add_argument("--out", default=None)
    ap.add_argument("--sections", default="A,B,C,D,E,F,G,H,I")
    ap.add_argument("--adam-steps", type=int, default=1000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    out = args.out or os.path.join(args.runs, "analysis_revision")
    os.makedirs(out, exist_ok=True)
    torch.set_grad_enabled(True)
    sec = set(args.sections.split(","))
    R = Report()
    if "A" in sec:
        hold = args.holdout if os.path.isdir(args.holdout) else None
        section_tf_timing(R, args.runs, hold, out)
        section_tf_matched(R, args.runs, hold)
    if "B" in sec:
        section_mlp_fit(R, args.runs, out)
    if "C" in sec:
        section_mlp_baselines(R, args.runs)
    if "D" in sec:
        section_tf_dense(R, args.dense if os.path.isdir(args.dense) else None, args.device, out)
    if "E" in sec:
        section_cka(R, args.runs)
    if "F" in sec:
        section_adam_state(R, args.runs, args.device, args.adam_steps)
    if "G" in sec:
        section_fine(R, args.runs, out)
    if "H" in sec:
        section_tf_caps(R, args.runs)
    if "I" in sec:
        section_synthetic(R, out)
    R.save(out)
    print(f"wrote {out}/summary.txt")


if __name__ == "__main__":
    main()
