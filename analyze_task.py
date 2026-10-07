"""
analyze_task.py -- the paper's key MLP results on a second task
(pre-registered hypotheses H2-H6 in PREREGISTRATION.md).

  python analyze_task.py --runs results --task mul     # multiplication mod 97
  python analyze_task.py --runs results --task p113    # addition mod 113

Reads <runs>/mlp/<task>_main (15 seeds) and <task>_rank_{128,64,32} (3 seeds each),
written by run_mlp.py. Writes <runs>/analysis_<task>/summary.txt, summary.json and
fig_dc_over_training.png. Definitions are imported from analyze_mlp.py / common_mlp.py,
so they are identical to the paper's.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats

from analyze_mlp import (Report, acc_with_W1, clean, d_at, fmt, load_exp, mstd, orth, savefig,
                         share_after_tg, steepest_drop, topk_curve)
from common_mlp import (compression_onset, effective_rank, final_model, grokking_onset,
                        model_from_weights, run_data)


def grokked(run):
    return float(run["hist"]["test_acc"][-1]) >= 0.95


def ttest(x, label):
    x = clean(x)
    if len(x) < 2:
        return f"{label}: fewer than two values, not tested", None
    t, p = stats.ttest_1samp(x, 0, alternative="greater")
    return (f"{label}: {fmt(x, 1)};  > 0 in {(x > 0).sum()}/{len(x)};  one-sided t={t:.2f}, p={p:.2e}  "
            f"-> {'supported' if p < 0.05 else 'not supported'} at alpha = 0.05"), float(p)


# ---------------------------------------------------------------- frequency usage
def primitive_root(p):
    n = p - 1
    qs = {q for q in range(2, n + 1) if n % q == 0 and all(q % d for d in range(2, int(q ** 0.5) + 1))}
    return next(g for g in range(2, p) if all(pow(g, n // q, p) != 1 for q in qs))


def freq_usage(W1, p, op):
    """(share of non-DC power in the top 5 frequencies, #frequencies for 90%, for 99%) of
    W1's operand-a block. For multiplication the nonzero residues are reordered by their
    discrete logarithm, so that a*b becomes addition mod p-1 and its Fourier basis applies."""
    M = W1[:, :p]
    if op == "mul":
        g = primitive_root(p)
        M = M[:, [pow(g, k, p) for k in range(p - 1)]]
    n = M.shape[1]
    pw = (np.abs(np.fft.fft(M, axis=1)) ** 2).mean(0)
    fs = np.arange(1, (n - 1) // 2 + 1)
    pr = list(pw[fs] + pw[n - fs]) + ([pw[n // 2]] if n % 2 == 0 else [])
    pr = np.sort(pr)[::-1]
    c = np.cumsum(pr) / pr.sum()
    return float(c[4]), int(np.argmax(c >= 0.90) + 1), int(np.argmax(c >= 0.99) + 1)


# ---------------------------------------------------------------- sections
def section_timing(R, runs):
    R.h("A. Timing of compression (H2, H3)")
    rows = []
    for r in runs:
        h, st = r["hist"], r["hist"]["steps"]
        tg = grokking_onset(st, h["test_acc"])
        tc = compression_onset(st, h["W1_erank"])
        ts = steepest_drop(st, h["W1_erank"], 1000)
        rows.append(dict(seed=r["seed"], tg=tg, tc=tc, ts=ts,
                         lag=None if tg is None or tc is None else tc - tg,
                         steep_lag=None if tg is None else ts - tg,
                         after=share_after_tg(st, h["W1_erank"], tg)))
        R.p(f"  seed {r['seed']:>2}: t_g={tg}  t_c={tc}  t_c-t_g={rows[-1]['lag']}  t_s={ts}  t_s-t_g={rows[-1]['steep_lag']}")
    R.p(f"  t_g {fmt([w['tg'] for w in rows], 0)};  t_c {fmt([w['tc'] for w in rows], 0)}")
    l2, p2 = ttest([w["lag"] for w in rows], "H2  t_c - t_g")
    l3, p3 = ttest([w["steep_lag"] for w in rows], "H3  t_s - t_g")
    R.p("  " + l2)
    R.p("  " + l3)
    R.p(f"  share of the ER(W1) decrease after t_g: {fmt([w['after'] for w in rows], 2)}")
    R.data["timing"] = dict(rows=rows, H2_p=p2, H3_p=p3)
    return {w["seed"]: w["tg"] for w in rows}


def section_dc(R, runs, tgs, out):
    R.h("B. Causal dimensionality over training (H4, H5) and end-of-training controls")
    table, per_seed = {}, []
    for r in runs:
        X, y, _, te = run_data(r)
        d_of = {}
        for s, w in sorted(r["ckpts"].items()):
            if s < 5000:
                continue
            m = model_from_weights(w["W1"], w["W2"], r["config"]["act"])
            ks, accs, orig, _ = topk_curve(m, X[te], y[te])
            d99 = d_at(ks, accs, orig, 0.99)
            d_of[s] = d99
            row = table.setdefault(s, dict(acc=[], d90=[], d99=[], er=[]))
            row["acc"].append(100 * orig)
            row["d90"].append(d_at(ks, accs, orig, 0.90))
            row["d99"].append(d99)
            row["er"].append(effective_rank(w["W1"]))
        tg, end = tgs[r["seed"]], r["final_step"]
        keys = sorted(d_of)
        s_g = next((s for s in keys if s >= tg), None) if tg is not None else None
        s_p = next((s for s in keys if s >= tg + 5000 and s != end), None) if tg is not None else None
        per_seed.append(dict(seed=r["seed"], s_g=s_g, s_plus=s_p, d_g=d_of.get(s_g), d_plus=d_of.get(s_p),
                             d_end=d_of.get(end)))
    R.p(f"  {'step':>7} {'test acc %':>20} {'d_0.9':>20} {'d_0.99':>20} {'ER(W1)':>20}")
    for s in sorted(table):
        d = table[s]
        R.p(f"  {s:>7} {fmt(d['acc'], 1):>20} {fmt(d['d90'], 1):>20} {fmt(d['d99'], 1):>20} {fmt(d['er'], 1):>20}")
    ok = [w for w in per_seed if w["s_plus"] is not None and w["d_g"] is not None and w["d_plus"] is not None]
    for w in per_seed:
        R.p(f"  seed {w['seed']:>2}: s_g={w['s_g']} (d={w['d_g']})  s_+={w['s_plus']} (d={w['d_plus']})  end d={w['d_end']}")
    l4, p4 = ttest([w["d_g"] - w["d_plus"] for w in ok], "H4  d(s_g) - d(s_+)")
    l5, p5 = ttest([w["d_end"] - w["d_plus"] for w in ok], "H5  d(s_end) - d(s_+)")
    R.p("  " + l4)
    R.p("  " + l5)
    R.data["dc"] = dict(table={str(k): v for k, v in table.items()}, per_seed=per_seed, H4_p=p4, H5_p=p5)

    # controls at the end of training, k = d_0.99 per seed (as in the paper's Table 2)
    ctrl = dict(top=[], bottom=[], subspace=[])
    for r in runs:
        X, y, _, te = run_data(r)
        model = final_model(r)
        ks, accs, orig, (U, S, Vh) = topk_curve(model, X[te], y[te])
        k = d_at(ks, accs, orig, 0.99)
        if k is None:
            continue
        n = len(S)
        rec = lambda idx: (U[:, idx] * S[idx]) @ Vh[idx]
        ctrl["top"].append(acc_with_W1(model, rec(list(range(k))), X[te], y[te]))
        ctrl["bottom"].append(acc_with_W1(model, rec(list(range(n - k, n))), X[te], y[te]))
        rng = np.random.default_rng(r["seed"])
        W1 = model.W1_full.detach()
        sub = []
        for _ in range(10):
            Q = torch.tensor(orth(rng.standard_normal((W1.shape[0], k))), dtype=W1.dtype)
            sub.append(acc_with_W1(model, Q @ (Q.T @ W1), X[te], y[te]))
        ctrl["subspace"].append(float(np.mean(sub)))
    chance = 100 / runs[0]["config"]["p"]
    R.p(f"\n  end of training, k = d_0.99 (test accuracy %, chance {chance:.2f}):  top {fmt([100 * v for v in ctrl['top']], 2)} | "
        f"bottom {fmt([100 * v for v in ctrl['bottom']], 2)} | random subspace {fmt([100 * v for v in ctrl['subspace']], 2)}")
    R.data["controls"] = ctrl

    st = sorted(table)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for key, lab in (("d99", "d_0.99"), ("d90", "d_0.9")):
        ax.errorbar(st, [mstd(table[s][key])[0] for s in st], yerr=[mstd(table[s][key])[1] for s in st],
                    marker="o", capsize=3, label=lab)
    ax.set_xscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("directions of W1 needed")
    ax2 = ax.twinx()
    ax2.plot(st, [mstd(table[s]["er"])[0] for s in st], "k--", label="ER(W1)")
    ax2.set_ylabel("effective rank")
    ax.legend(loc="upper right", fontsize=8)
    ax2.legend(loc="lower right", fontsize=8)
    savefig(out, "fig_dc_over_training.png")


def section_factorization(R, root, task, main):
    R.h("C. Rank-constrained parameterization W1 = AB (H6, descriptive)")
    groups = [("W1, unfactorized (main, seeds 0-2)", [r for r in main if r["seed"] < 3])] + \
             [(f"AB, rank {k}" + (" (full-rank control)" if k == 128 else ""), load_exp(root, f"{task}_rank_{k}"))
              for k in (128, 64, 32)]
    res = {}
    for name, rs in groups:
        if not rs:
            R.p(f"  {name:<36} no runs")
            continue
        tg = [grokking_onset(r["hist"]["steps"], r["hist"]["test_acc"]) for r in rs]
        R.p(f"  {name:<36} grokked {len(clean(tg))}/{len(rs)}  t_g {fmt(tg, 0)}  "
            f"final test {fmt([100 * r['hist']['test_acc'][-1] for r in rs], 1)}%  "
            f"final ER {fmt([r['hist']['W1_erank'][-1] for r in rs], 1)}")
        res[name] = tg
    R.data["factorization"] = res


def section_freqs(R, runs, dc_end):
    R.h("D. Frequency usage of W1 (descriptive)")
    op, p = runs[0]["config"]["op"], runs[0]["config"]["p"]
    if op == "mul":
        R.p(f"  multiplication: nonzero residues ordered by discrete log base {primitive_root(p)}, "
            f"Fourier transform over Z_{p - 1}")
    use = [freq_usage(r["ckpts"][r["final_step"]]["W1"].numpy(), p, op) for r in runs]
    R.p(f"  top-5 share {fmt([u[0] for u in use], 2)};  #freq for 90% {fmt([u[1] for u in use], 1)};  "
        f"for 99% {fmt([u[2] for u in use], 1)}")
    R.p(f"  2 x #freq (90%) = {fmt([2 * u[1] for u in use], 1)};  2 x #freq (99%) = {fmt([2 * u[2] for u in use], 1)};  "
        f"d_0.99 at the end = {fmt(dc_end, 1)}")
    R.data["freq_usage"] = use


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results")
    ap.add_argument("--task", required=True, help="experiment tag in run_mlp.py, e.g. mul or p113")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(args.runs, f"analysis_{args.task}")
    os.makedirs(out, exist_ok=True)
    torch.set_grad_enabled(False)

    main_all = load_exp(args.runs, f"{args.task}_main")
    if not main_all:
        raise SystemExit(f"no runs in {args.runs}/mlp/{args.task}_main")
    runs = [r for r in main_all if grokked(r)]
    R = Report()
    c = main_all[0]["config"]
    R.p(f"task {args.task}: op={c['op']}, p={c['p']}; {len(main_all)} main seeds, {len(runs)} grokked "
        f"(final test accuracy >= 95%) and used for H2-H5")
    for r in main_all:
        if not grokked(r):
            R.p(f"  seed {r['seed']} excluded: final test accuracy {100 * r['hist']['test_acc'][-1]:.1f}%")
    if len(runs) >= 2:
        tgs = section_timing(R, runs)
        section_dc(R, runs, tgs, out)
        section_freqs(R, runs, [w["d_end"] for w in R.data["dc"]["per_seed"]])
    else:
        R.p("fewer than two grokked seeds: H2-H5 not tested")
    section_factorization(R, args.runs, args.task, main_all)
    R.save(out)
    print("\n".join(R.lines))
    print(f"\nwrote {out}/summary.txt, summary.json and figures")


if __name__ == "__main__":
    main()
