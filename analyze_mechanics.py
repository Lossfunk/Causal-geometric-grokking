"""
analyze_mechanics.py -- why the weight norm grows in late training despite weight decay.

  python analyze_mechanics.py --runs results --out results/analysis_mechanics

Reads the experiments added to run_mlp.py for this question and writes summary.txt,
summary.json and figures. Sections run only if their experiments exist.

  A  Optimizer swap after grokking (cont_*): main seeds continued from step 20,000 with
     AdamW, SGD at the same per-step decay, AdamW without decay, SGD (lr 1) without decay.
  B  Radial forces (forces_wd*): per logged step, <theta, delta theta> split into the
     weight-decay part (-lr * wd * ||theta||^2) and the loss-driven (optimizer) part.
  C  Long runs (long_*): does ||W1|| settle at an equilibrium that depends on wd?
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from analyze_mlp import Report, d99_at, fmt, load_exp, savefig
from common_mlp import effective_rank, grokking_onset, run_data, model_from_weights, accuracy


STAT_KEYS = ("W1", "W2", "s1", "er", "d99", "acc")


def diverged(r, s):
    w = r["ckpts"][s]
    return not (torch.isfinite(w["W1"]).all() and torch.isfinite(w["W2"]).all())


def ckpt_stats(r, s):
    w = r["ckpts"][s]
    if diverged(r, s):                     # weights overflowed (too large a learning rate)
        return dict.fromkeys(STAT_KEYS)
    X, y, _, te = run_data(r)
    m = model_from_weights(w["W1"], w["W2"], r["config"]["act"])
    with torch.no_grad():
        acc = accuracy(m(X[te]), y[te])
    return dict(W1=float(w["W1"].norm()), W2=float(w["W2"].norm()),
                s1=float(torch.linalg.svdvals(w["W1"].float())[0]),
                er=effective_rank(w["W1"]), d99=d99_at(r, s), acc=100 * acc)


def forces(h):
    """Radial parts per logged step: optimizer (loss-driven) and weight decay."""
    upd, dec = np.asarray(h["radial_update"], float), np.asarray(h["radial_decay"], float)
    return upd - dec, dec


def section_continuation(R, root, out):
    exps = [("cont_adamw", "AdamW (control)"), ("cont_sgd", "SGD, same decay"),
            ("cont_adamw_wd0", "AdamW, no decay"), ("cont_sgd_lr1_wd0", "SGD lr 1, no decay")]
    exps += [(f"cont_gd_lr{lr:g}", f"GD lr {lr:g}") for lr in (1e2, 1e3, 1e4, 1e5, 1e6)]
    exps += [(f"cont_sgdm_lr{lr:g}", f"SGD+mom. lr {lr:g}") for lr in (1e1, 1e2, 1e3, 1e4, 1e5)]
    exps += [(f"cont_adamw_eps{e:g}", f"AdamW eps {e:g}") for e in (1e-10, 1e-9, 1e-7, 1e-6, 1e-4)]
    exps = [(e, lab, load_exp(root, e)) for e, lab in exps]
    exps = [x for x in exps if x[2]]
    if not exps:
        return
    R.h("A. Optimizer swap after grokking: main seeds continued from step 20,000")
    base = {r["seed"]: r for r in load_exp(root, "main")}
    keys = [("W1", "||W1||", 0), ("W2", "||W2||", 0), ("s1", "sigma_1(W1)", 1),
            ("er", "ER(W1)", 1), ("d99", "d_0.99", 1), ("acc", "test acc %", 1)]
    data = {}
    rows = [("main (original run)", [base[r["seed"]] for r in exps[0][2] if r["seed"] in base])] + \
           [(lab, rs) for _, lab, rs in exps]
    for lab, rs in rows:
        if not rs:
            continue
        start = exps[0][2][0]["config"]["init_step"]      # the step the continuations start from
        steps = sorted(s for s in rs[0]["ckpts"] if s >= start)
        n_div = sum(diverged(r, r["final_step"]) for r in rs)
        R.p(f"\n  [{lab}]  n={len(rs)}" + (f"  DIVERGED in {n_div}/{len(rs)} seeds (non-finite weights; "
                                            f"excluded from the statistics below)" if n_div else ""))
        R.p("  " + f"{'step':>8}" + "".join(f"{k[1]:>22}" for k in keys))
        data[lab] = {}
        for s in steps:
            st = [ckpt_stats(r, s) for r in rs if s in r["ckpts"]]
            data[lab][s] = st
            R.p("  " + f"{s:>8}" + "".join(f"{fmt([x[k] for x in st], nd):>22}" for k, _, nd in keys))
        ok = [r for r in rs if not diverged(r, r["final_step"])]
        if ok and "radial_update" in ok[0]["hist"] and len(ok[0]["hist"]["radial_update"]):
            opt_part = [forces(r["hist"])[0].mean() for r in ok]
            dec_part = [forces(r["hist"])[1].mean() for r in ok]
            R.p(f"  mean radial force per logged step: optimizer {fmt(opt_part, 4)}   weight decay {fmt(dec_part, 4)}")
    R.data["continuation"] = data
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    for lab, rs in rows:
        if not rs:
            continue
        h = rs[0]["hist"]
        st = np.asarray(h["steps"])
        sel = st >= exps[0][2][0]["config"]["init_step"]
        axs[0].plot(st[sel], np.asarray(h["W1_norm"])[sel], label=lab)
        axs[1].plot(st[sel], np.asarray(h["W1_erank"])[sel], label=lab)
    axs[0].set_ylabel("||W1|| (seed 0)")
    axs[1].set_ylabel("effective rank of W1 (seed 0)")
    for ax in axs:
        ax.set_xlabel("step")
        ax.legend(fontsize=7)
    savefig(out, "fig_optimizer_swap.png")


def section_forces(R, root, out):
    exps = [(wd, load_exp(root, f"forces_wd{wd:g}")) for wd in (0.0, 1e-2, 1e-1)]
    exps = [(wd, rs) for wd, rs in exps if rs]
    if not exps:
        return
    R.h("B. Radial forces on the weight norm (optimizer part vs weight-decay part)")
    R.p("  per logged step: optimizer part = <theta, delta theta> - weight-decay part;")
    R.p("  the norm grows when the optimizer part exceeds |weight-decay part|")
    R.p(f"  {'weight decay':<13} {'window':<24} {'optimizer part':>22} {'decay part':>22} {'||W1|| change':>16}")
    data = {}
    for wd, rs in exps:
        data[str(wd)] = {}
        for lab, lo, hi in (("t_g .. t_g+10k", None, 10_000), ("40k .. 60k", 40_000, 60_000),
                            ("80k .. 100k", 80_000, 100_000)):
            o, d, dn = [], [], []
            for r in rs:
                h = r["hist"]
                st = np.asarray(h["steps"])
                tg = grokking_onset(h["steps"], h["test_acc"]) or 0
                a, b = (tg, tg + hi) if lo is None else (lo, hi)
                sel = (st >= a) & (st < b)
                if not sel.any():
                    continue
                op, de = forces(h)
                o.append(op[sel].mean())
                d.append(de[sel].mean())
                n = np.asarray(h["W1_norm"])[sel]
                dn.append(n[-1] - n[0])
            data[str(wd)][lab] = dict(optimizer=o, decay=d, dnorm=dn)
            R.p(f"  {wd:<13g} {lab:<24} {fmt(o, 4):>22} {fmt(d, 4):>22} {fmt(dn, 1):>16}")
    R.data["forces"] = data
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for wd, rs in exps:
        h = rs[0]["hist"]
        st = np.asarray(h["steps"])
        op, de = forces(h)
        line, = ax.plot(st, op, lw=0.9, label=f"optimizer part, wd {wd:g}")
        if wd > 0:
            ax.plot(st, -de, lw=0.9, ls="--", color=line.get_color(), label=f"|decay part|, wd {wd:g}")
    ax.set_yscale("symlog", linthresh=1e-4)
    ax.set_xlabel("step")
    ax.set_ylabel("radial force per step (seed 0)")
    ax.legend(fontsize=7)
    savefig(out, "fig_radial_forces.png")


def section_long(R, root, out):
    exps = [(e, load_exp(root, e)) for e in ("long_mse", "long_mse_wd0.03", "long_ce_lr0.01")]
    exps = [(e, rs) for e, rs in exps if rs]
    if not exps:
        return
    R.h("C. Long runs: does ||W1|| settle at an equilibrium?")
    data = {}
    for e, rs in exps:
        steps = sorted(s for s in rs[0]["ckpts"] if s >= 100_000) or sorted(rs[0]["ckpts"])
        R.p(f"\n  [{e}]  n={len(rs)}")
        R.p(f"  {'step':>8} {'||W1||':>20} {'change since prev.':>20} {'ER(W1)':>20} {'d_0.99':>20} {'test acc %':>20}")
        prev, data[e] = None, {}
        for s in steps:
            st = [ckpt_stats(r, s) for r in rs if s in r["ckpts"]]
            data[e][s] = st
            nrm = [x["W1"] for x in st]
            ch = "" if prev is None else fmt([a - b for a, b in zip(nrm, prev)], 1)
            R.p(f"  {s:>8} {fmt(nrm, 1):>20} {ch:>20} {fmt([x['er'] for x in st], 1):>20} "
                f"{fmt([x['d99'] for x in st], 1):>20} {fmt([x['acc'] for x in st], 1):>20}")
            prev = nrm
    R.data["long"] = data
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for e, rs in exps:
        h = rs[0]["hist"]
        ax.plot(h["steps"], h["W1_norm"], label=e)
    ax.set_xscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("||W1|| (seed 0)")
    ax.legend(fontsize=7)
    savefig(out, "fig_long_runs.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(args.runs, "analysis_mechanics")
    os.makedirs(out, exist_ok=True)
    torch.set_grad_enabled(False)
    R = Report()
    section_continuation(R, args.runs, out)
    section_forces(R, args.runs, out)
    section_long(R, args.runs, out)
    if not R.lines:
        raise SystemExit(f"no cont_*, forces_* or long_* runs in {args.runs}/mlp")
    R.save(out)
    print(f"wrote {out}/summary.txt")


if __name__ == "__main__":
    main()
