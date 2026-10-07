"""
analyze_transformer.py -- every transformer table and figure, from saved runs.

  python analyze_transformer.py --runs results --out results/analysis_transformer

Reads <runs>/transformer/wd*/seed*.pt (written by run_transformer.py).
Main analyses use the wd=1 runs that grokked (final test accuracy >= 95%).
Runs that did not grok (any wd) are reported as negative controls (section N)
and give the false-alarm counts of the IPR detectors (section F); other wd
folders also feed the alignment-vs-wd comparison (section I).
Writes summary.txt, summary.json and fig_*.png.
Requires: pip install transformer_lens
"""
import argparse
import glob
import json
import os
import re
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats
from scipy.signal import correlate

from run_transformer import MATS, P, as2d, build_dataset, build_model, weight_matrices

THRESHOLDS = (0.90, 0.99)


class Report:
    def __init__(self):
        self.lines, self.data = [], {}

    def h(self, t):
        self.lines += ["", "=" * 76, t, "=" * 76]
        print(f"\n== {t}", flush=True)

    def p(self, *a):
        self.lines.append(" ".join(str(x) for x in a))

    def save(self, out):
        with open(os.path.join(out, "summary.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(self.lines) + "\n")
        with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


def clean(xs):
    return np.asarray([x for x in xs if x is not None], dtype=float)


def fmt(xs, nd=1, med=False):
    x = clean(xs)
    if not len(x):
        return "n/a"
    sd = x.std(ddof=1) if len(x) > 1 else 0.0
    m = f", median {np.median(x):.{nd}f}" if med else ""
    return f"{x.mean():.{nd}f} ± {sd:.{nd}f} (n={len(x)}{m})"


def t_grok(h, thr=0.95):
    return next((int(s) for s, a in zip(h["steps"], h["test_acc"]) if a >= thr), None)


def grokked(run, thr=0.95):
    """Final test accuracy >= 95%. Runs that miss this within the step budget
    are kept out of the main tables and used as negative controls."""
    return float(run["hist"]["test_acc"][-1]) >= thr


def share_after_tg(steps, er, tg):
    """Share of the decrease in ER, from its maximum up to t_g down to its
    final value, that happens after t_g (1 = all of it, 0 = none of it).
    Same definition in analyze_mlp.py."""
    if tg is None:
        return None
    steps, er = np.asarray(steps), np.asarray(er, dtype=float)
    peak, at, end = er[steps <= tg].max(), er[steps == tg][0], er[-1]
    return float((at - end) / (peak - end)) if peak > end else None


def freq_usage(M, p=P):
    """How many frequencies a weight matrix uses. M has one row per unit and p
    columns (token or class axis). Returns (share of non-DC power in the top 5
    frequencies, #frequencies for 90% of it, #frequencies for 99%), with f and
    p-f counted as one frequency. Same function in analyze_mlp.py."""
    pw = (np.abs(np.fft.fft(M, axis=1)) ** 2).mean(0)
    fs = np.arange(1, (p - 1) // 2 + 1)
    pr = np.sort(pw[fs] + pw[p - fs])[::-1]
    c = np.cumsum(pr) / pr.sum()
    return float(c[4]), int(np.argmax(c >= 0.90) + 1), int(np.argmax(c >= 0.99) + 1)


def orth(M):
    return np.linalg.qr(M)[0]


def subspace_cos(A, B):
    return float(np.linalg.svd(A.T @ B, compute_uv=False).mean())


def random_baseline(k, d, n=300, seed=0):
    rng = np.random.default_rng(seed)
    v = [subspace_cos(orth(rng.standard_normal((d, k))), orth(rng.standard_normal((d, k)))) for _ in range(n)]
    return float(np.mean(v)), float(np.std(v))


def savefig(out, name):
    plt.tight_layout()
    plt.savefig(os.path.join(out, name), dpi=200)
    plt.close()


def load_runs(root, wd):
    paths = sorted(glob.glob(os.path.join(root, "transformer", f"wd{wd:g}", "seed*.pt")),
                   key=lambda p: int(re.findall(r"seed(\d+)", p)[0]))
    return [torch.load(p, map_location="cpu", weights_only=False) for p in paths]


def all_wds(root):
    return sorted({float(re.findall(r"wd([0-9.e-]+)$", p)[0]) for p in
                   glob.glob(os.path.join(root, "transformer", "wd*"))})


def restore(run, device, step=None):
    model = build_model(run["seed"], device)
    model.load_state_dict(run["ckpts"][run["final_step"] if step is None else step])
    model.eval()
    data, labels, tr, te = build_dataset(run["seed"], run["config"]["frac"], device)
    return model, data, labels, tr, te


def acc(model, data, labels):
    with torch.no_grad():
        return float((model(data)[:, -1, :].argmax(-1) == labels).float().mean())


def set_weight(model, name, W):
    blk = model.blocks[0]
    target = {"W_E": model.embed.W_E[:-1], "W_U": model.unembed.W_U, "W_in": blk.mlp.W_in,
              "W_out": blk.mlp.W_out, "W_Q": blk.attn.W_Q, "W_K": blk.attn.W_K,
              "W_V": blk.attn.W_V, "W_O": blk.attn.W_O}[name]
    with torch.no_grad():
        target.copy_(W)


def truncated(W, idx):
    """Rebuild W from the singular triplets in idx (3-D head tensors are
    flattened to (-1, last) as in the notebook, i.e. heads truncated jointly)."""
    W2 = as2d(W).float()
    U, S, Vh = torch.linalg.svd(W2, full_matrices=False)
    idx = torch.as_tensor(list(idx))
    return ((U[:, idx] * S[idx]) @ Vh[idx]).reshape(W.shape), S


# ---------------------------------------------------------------- A. spectra
def k_at(frac_cum, q):
    return int(np.argmax(frac_cum >= q) + 1)


def section_spectra(R, runs, device, out):
    R.h("A. Spectra and variance thresholds (Fig. 8)")
    rows = {m: {"var95": [], "var99": [], "sv95": [], "sv99": []} for m in MATS + ["MLP post-act"]}
    spectra0 = {}
    for r in runs:
        model, data, labels, tr, te = restore(r, device)
        wm = weight_matrices(model)
        for m in MATS:
            s = torch.linalg.svdvals(as2d(wm[m]).detach().float()).cpu().numpy()
            v = np.cumsum(s ** 2) / (s ** 2).sum()
            u = np.cumsum(s) / s.sum()
            rows[m]["var95"].append(k_at(v, .95)); rows[m]["var99"].append(k_at(v, .99))
            rows[m]["sv95"].append(k_at(u, .95)); rows[m]["sv99"].append(k_at(u, .99))
            if r is runs[0]:
                spectra0[m] = s
        with torch.no_grad():
            _, cache = model.run_with_cache(data)
        Z = cache["post", 0, "mlp"][:, -1, :].cpu().numpy()
        S = np.linalg.svd(Z - Z.mean(0), compute_uv=False)
        v = np.cumsum(S ** 2) / (S ** 2).sum()
        rows["MLP post-act"]["var95"].append(k_at(v, .95)); rows["MLP post-act"]["var99"].append(k_at(v, .99))
        if r is runs[0]:
            spectra0["MLP post-act"] = S
    R.p(f"  {'matrix':<14} {'k95 (variance)':>32} {'k99 (variance)':>32} {'k95 (sum of sv)':>32}")
    for m, d in rows.items():
        R.p(f"  {m:<14} {fmt(d['var95'], med=True):>32} {fmt(d['var99'], med=True):>32} "
            f"{fmt(d['sv95'], med=True) if d['sv95'] else '-':>32}")
    R.data["spectra"] = rows
    names = list(spectra0)
    fig, axs = plt.subplots(1, len(names), figsize=(3 * len(names), 3))
    for ax, m in zip(axs, names):
        s = spectra0[m]
        v = np.cumsum(s ** 2) / (s ** 2).sum()
        ax.semilogy(np.arange(1, len(s) + 1), s, ".-", ms=3)
        for q, col in ((.95, "orange"), (.99, "red")):
            k = k_at(v, q)
            ax.axvline(k, color=col, ls="--", lw=1, label=f"{int(q * 100)}% var: {k}")
        ax.set_title(m, fontsize=9)
        ax.legend(fontsize=7)
    savefig(out, "fig_transformer_spectra.png")


# ---------------------------------------------------------------- B/C. causal dims + controls
def weight_curve(model, name, data, labels):
    W0 = weight_matrices(model)[name].detach().clone()
    n = int(torch.linalg.svdvals(as2d(W0).float()).shape[0])
    ks = sorted(set(range(1, min(21, n + 1))) | set(range(25, n + 1, 5)) | {n})   # always test full rank
    accs = []
    for k in ks:
        Wk, _ = truncated(W0, range(k))
        set_weight(model, name, Wk)
        accs.append(acc(model, data, labels))
    set_weight(model, name, W0)
    return ks, accs, n


def first_k(ks, accs, orig, tau):
    for k, a in zip(ks, accs):
        if a >= tau * orig:
            return k
    return None


def activation_dc(model, data, labels):
    with torch.no_grad():
        logits, cache = model.run_with_cache(data)
        orig = float((logits[:, -1, :].argmax(-1) == labels).float().mean())
        Z = cache["post", 0, "mlp"][:, -1, :]
        mu = Z.mean(0)
        U, S, Vh = torch.linalg.svd(Z - mu, full_matrices=False)
        mid = cache["resid_mid", 0][:, -1, :]
        blk = model.blocks[0]
        for k in range(1, Z.shape[1] + 1):
            Zk = (U[:, :k] * S[:k]) @ Vh[:k] + mu
            out = model.unembed((Zk @ blk.mlp.W_out + blk.mlp.b_out + mid)[:, None, :])[:, 0, :]
            if float((out.argmax(-1) == labels).float().mean()) >= 0.99 * orig:
                return k
    return None


def section_causal_dims(R, runs, device):
    R.h("B. Accuracy-preserving rank d_c (Table 6) and C. truncation controls (Table 2)")
    dc = {m: [] for m in MATS + ["MLP post-act"]}
    ctrl = {m: {"top": [], "bottom": [], "rand_subset": [], "rand_subspace": []}
            for m in ("W_E", "W_U", "W_in", "W_out")}
    nmax = {}
    for r in runs:
        model, data, labels, tr, te = restore(r, device)
        Xte, yte = data[te], labels[te]
        orig = acc(model, Xte, yte)
        for m in MATS:
            ks, accs, n = weight_curve(model, m, Xte, yte)
            nmax[m] = n
            dc[m].append(first_k(ks, accs, orig, 0.99))
        dc["MLP post-act"].append(activation_dc(model, Xte, yte))
        nmax["MLP post-act"] = 512
        rng = np.random.default_rng(r["seed"])
        for m in ctrl:
            k = dc[m][-1]
            if k is None:
                continue
            W0 = weight_matrices(model)[m].detach().clone()
            n = nmax[m]
            for key, idxs in (("top", [range(k)]), ("bottom", [range(n - k, n)]),
                              ("rand_subset", [sorted(rng.choice(n, k, replace=False)) for _ in range(10)])):
                vals = []
                for idx in idxs:
                    Wk, _ = truncated(W0, [int(i) for i in idx])
                    set_weight(model, m, Wk)
                    vals.append(acc(model, Xte, yte))
                ctrl[m][key].append(float(np.mean(vals)))
            W2 = as2d(W0).float()
            vals = []
            for _ in range(10):
                Q = torch.tensor(orth(rng.standard_normal((W2.shape[0], k))), dtype=W2.dtype, device=W2.device)
                set_weight(model, m, (Q @ (Q.T @ W2)).reshape(W0.shape))
                vals.append(acc(model, Xte, yte))
            ctrl[m]["rand_subspace"].append(float(np.mean(vals)))
            set_weight(model, m, W0)
    R.p(f"  {'matrix':<14} {'max rank':>9} {'d_c (99% of clean acc)':>34}")
    for m in dc:
        R.p(f"  {m:<14} {nmax.get(m, '?'):>9} {fmt(dc[m], med=True):>34}")
    R.p("\n  controls at k = d_c (test accuracy, %):")
    for m, d in ctrl.items():
        R.p(f"  {m:<6} top {fmt([100 * v for v in d['top']], 2)} | bottom {fmt([100 * v for v in d['bottom']], 2)} | "
            f"random subset {fmt([100 * v for v in d['rand_subset']], 2)} | random subspace "
            f"{fmt([100 * v for v in d['rand_subspace']], 2)}")
    R.data["d_c"] = dc
    R.data["controls"] = ctrl
    return dc


# ---------------------------------------------------------------- D. read/write alignment
def key_freq_subspace(W_E, K=5):
    E = W_E.detach().float().cpu().numpy()                  # (97, 128)
    n = np.arange(P)
    dirs, power = {}, {}
    for f in range(1, (P - 1) // 2 + 1):
        c, s = E.T @ np.cos(2 * np.pi * f * n / P), E.T @ np.sin(2 * np.pi * f * n / P)
        dirs[f], power[f] = (c, s), float(c @ c + s @ s)
    top = sorted(power, key=power.get, reverse=True)[:K]
    M = np.stack([v for f in top for v in dirs[f]] + [E.T @ np.ones(P)], axis=1)
    return orth(M), top


def read_space(W, k):
    """Top-k left singular vectors of a (d_model, x) matrix: what it reads."""
    return torch.linalg.svd(as2d(W).detach().float(), full_matrices=False)[0][:, :k].cpu().numpy()


def write_space(W, k):
    """Top-k right singular vectors of an (x, d_model) matrix: what it writes."""
    return torch.linalg.svd(as2d(W).detach().float(), full_matrices=False)[2][:k].T.cpu().numpy()


ALIGN_PAIRS = {
    "attn_write_vs_mlp_read": "attention write (W_O) vs MLP read (W_in)",
    "mlp_read_vs_write": "MLP read (W_in) vs MLP write (W_out)",
    "mlp_write_vs_unembed_read": "MLP write (W_out) vs unembed read (W_U)",
    "mlp_read_vs_keyfreq": "MLP read (W_in) vs key-frequency subspace of W_E",
    "mlp_write_vs_keyfreq": "MLP write (W_out) vs key-frequency subspace of W_E",
}


def alignment(model, k=11):
    """Mean principal-angle cosine between k-dim subspaces of the residual
    stream (R^128) along the path embed -> attention -> MLP -> unembed."""
    wm = weight_matrices(model)
    mr, mw = read_space(wm["W_in"], k), write_space(wm["W_out"], k)
    aw, ur = write_space(wm["W_O"], k), read_space(wm["W_U"], k)
    Kf, top = key_freq_subspace(wm["W_E"])
    return dict(attn_write_vs_mlp_read=subspace_cos(aw, mr),
                mlp_read_vs_write=subspace_cos(mr, mw),
                mlp_write_vs_unembed_read=subspace_cos(mw, ur),
                mlp_read_vs_keyfreq=subspace_cos(mr, Kf),
                mlp_write_vs_keyfreq=subspace_cos(mw, Kf)), top


def section_alignment(R, runs, device):
    R.h("D. Subspace alignment along the residual stream (k = 11 = 2 x 5 key frequencies + DC)")
    vals = {key: [] for key in ALIGN_PAIRS}
    for r in runs:
        a, _ = alignment(restore(r, device)[0])
        for key in ALIGN_PAIRS:
            vals[key].append(a[key])
    base = random_baseline(11, 128)
    R.p(f"  random baseline (two random 11-dim subspaces of R^128): {base[0]:.3f} ± {base[1]:.3f}")
    for key, lab in ALIGN_PAIRS.items():
        R.p(f"  {lab:<52} {fmt(vals[key], 3)}")
    R.p(f"  key frequencies of W_E (seed {runs[0]['seed']}): {alignment(restore(runs[0], device)[0])[1]}")
    R.data["alignment"] = dict(vals, random=base)


def section_alignment_vs_wd(R, root, device):
    wds = all_wds(root)
    if len(wds) < 2:
        return
    R.h("I. Alignment vs weight decay and grokking (T3), k = 11")
    R.p(f"  random baseline: {random_baseline(11, 128)[0]:.3f}")
    res = {}
    for wd in wds:
        runs = load_runs(root, wd)
        for ok, lab in ((True, "grokked"), (False, "not grokked")):
            rs = [r for r in runs if grokked(r) == ok]
            if not rs:
                continue
            a = [alignment(restore(r, device)[0])[0] for r in rs]
            res[f"wd={wd:g}|{lab}"] = a
            R.p(f"  wd={wd:<5g} {lab:<12} n={len(rs):<3} MLP read/write {fmt([x['mlp_read_vs_write'] for x in a], 3)}"
                f"   attention write/MLP read {fmt([x['attn_write_vs_mlp_read'] for x in a], 3)}"
                f"   t_g {fmt([t_grok(r['hist']) for r in rs], 0)}")
    R.data["alignment_vs_wd"] = res


# ---------------------------------------------------------------- E/F. timing
def onset(steps, vals, window_end, k, direction="up"):
    steps, vals = np.asarray(steps), np.asarray(vals, float)
    base = vals[steps < window_end]
    if len(base) < 2:
        return None
    mu, sd = base.mean(), base.std() + 1e-10
    for s, v in zip(steps, vals):
        if s >= window_end and ((direction == "up" and v > mu + k * sd) or
                                (direction == "down" and v < mu - k * sd)):
            return int(s)
    return None


def compression_onset(steps, er, min_step=1000, drop=0.01):
    run = er[0]
    for s, e in zip(steps, er):
        run = max(run, e)
        if s >= min_step and e < (1 - drop) * run:
            return int(s)
    return None


def steepest_drop(steps, er, start, smooth=5):
    """Step of the steepest decrease of ER (moving average over `smooth` logged
    points), ignoring steps before `start` and the last 4 logged steps.
    Same definition in analyze_mlp.py (there with start=1000)."""
    st, er = np.asarray(steps), np.asarray(er, dtype=float)
    d = np.gradient(np.convolve(er, np.ones(smooth) / smooth, mode="same"), st)
    ok = (st >= start) & (st <= st[-1] - 4 * (st[1] - st[0]))
    return int(st[ok][np.argmin(d[ok])])


def first_logged(steps, s0):
    st = np.asarray(steps)
    e = st[st >= s0]
    return int(e[0]) if len(e) else None


def section_timing(R, runs, neg, out):
    R.h("E. Timing of compression in the transformer (Table 1)")
    rows = []
    for r in runs:
        h = r["hist"]
        st = h["steps"]
        tg = t_grok(h)
        tc = compression_onset(st, h["W_in_erank"])
        rows.append(dict(seed=r["seed"], tg=tg, tc=tc, lag=None if tg is None or tc is None else tc - tg,
                         after={m: share_after_tg(st, h[f"{m}_erank"], tg) for m in MATS}))
    for w in rows:
        a = w["after"]["W_in"]
        a = "n/a" if a is None else f"{a:.2f}"
        R.p(f"  seed {w['seed']:>2}: t_g={w['tg']}  t_c(W_in)={w['tc']}  lag={w['lag']}  "
            f"share of ER(W_in) decrease after t_g={a}")
    lags = clean([w["lag"] for w in rows])
    R.p(f"  t_g {fmt([w['tg'] for w in rows], 0)};  t_c {fmt([w['tc'] for w in rows], 0)};  lag {fmt(lags, 0)}")
    n_first = sum(w["tc"] is not None and w["tc"] == first_logged(r["hist"]["steps"], 1000)
                  for w, r in zip(rows, runs))
    R.p(f"  the MLP's t_c rule (ER 1% below its running max, from step 1000) fires at its first eligible")
    R.p(f"  step in {n_first}/{len(rows)} seeds: with wd=1 the random initialisation decays from step 0, so")
    R.p(f"  ER falls from the start and this rule does not mark a compression event here.")
    if len(lags) > 1:
        t, pv = stats.ttest_1samp(lags, 0, alternative="greater")
        R.p(f"  t_c > t_g in {(lags > 0).sum()}/{len(lags)} seeds; one-sided t={t:.2f}, p={pv:.2e}")
    R.p("\n  share of each matrix's ER decrease (max up to t_g -> final) that happens after t_g:")
    for m in MATS:
        R.p(f"    {m:<6} {fmt([w['after'][m] for w in rows], 2, med=True)}")
    # the decay of the initialisation dominates ER early (factor 0.999 per step at lr 1e-3, wd 1);
    # by step 4,000 it is down to 2%, so look for the steepest decrease after that
    sl = clean([steepest_drop(r["hist"]["steps"], r["hist"]["W_in_erank"], 4000) - w["tg"]
                for r, w in zip(runs, rows) if w["tg"] is not None])
    if len(sl) > 1:
        t, pv = stats.ttest_1samp(sl, 0, alternative="greater")
        R.p(f"\n  steepest ER(W_in) decrease (smoothed, from step 4,000) minus t_g: {fmt(sl, 0, med=True)};"
            f"  after t_g in {(sl > 0).sum()}/{len(sl)};  one-sided t={t:.2f}, p={pv:.1e}"
            f"  (same measure in analyze_mlp.py section A)")
        R.data["steepest_drop_lag"] = sl

    le = runs[0]["config"]["log_every"]
    xs = []
    for r in runs:
        h = r["hist"]
        de, da = np.gradient(h["W_in_erank"]), np.gradient(h["test_acc"])
        de, da = (de - de.mean()) / (de.std() + 1e-10), (da - da.mean()) / (da.std() + 1e-10)
        xs.append(correlate(de, da, mode="full") / len(de))
    n = min(len(x) for x in xs)
    lags_s = (np.arange(n) - (n - 1) // 2) * le
    per_seed = [int(lags_s[np.argmin(x[:n])]) for x in xs]
    R.p(f"\n  cross-correlation of dER(W_in)/dt and dAcc/dt (tau > 0: rank changes lag accuracy changes)")
    R.p(f"  trough of the mean curve at tau = {int(lags_s[np.argmin(np.mean([x[:n] for x in xs], 0))]):+d} steps;"
        f"  per-seed troughs {fmt(per_seed, 0, med=True)}")
    R.data["timing"] = dict(rows=rows, xcorr_per_seed=per_seed)

    # ER and accuracy aligned at t_g, one line per seed
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), sharex=True)
    for r, w in zip(runs, rows):
        if w["tg"] is None:
            continue
        h = r["hist"]
        x = np.asarray(h["steps"]) - w["tg"]
        axs[0].plot(x, np.asarray(h["W_in_erank"]) / h["W_in_erank"][0], lw=0.8, alpha=0.7)
        axs[1].plot(x, h["test_acc"], lw=0.8, alpha=0.7)
    for ax, lab in zip(axs, ("ER(W_in) / ER at step 0", "test accuracy")):
        ax.axvline(0, color="k", lw=0.8, ls="--")
        ax.set_xlabel("step - t_g")
        ax.set_ylabel(lab)
    savefig(out, "fig_transformer_er_aligned.png")

    R.h("F. IPR onset sensitivity (Table 4)")
    R.p("  early = fires before t_g, out of the seeds with t_g > window (only those can be early);")
    R.p("  at start = fires at the first logged step after the window; false alarms = non-grokked")
    R.p(f"  runs (see section N) where it fires: {len(neg)} runs.")
    res = {}
    for metric in ("W_in_spectral_ipr", "E_fourier_ipr"):
        R.p(f"\n  metric={metric}")
        R.p(f"    {'window':>8} {'k':>3} {'detected':>9} {'early':>7} {'at start':>9} {'median lead':>12} "
            f"{'false alarms':>13}")
        for W in (2500, 5000, 10000):
            for k in (2, 3, 4):
                leads, ons, first = [], [], 0
                for r, w in zip(runs, rows):
                    o = onset(r["hist"]["steps"], r["hist"][metric], W, k)
                    ons.append(o)
                    first += int(o is not None and o == first_logged(r["hist"]["steps"], W))
                    leads.append(None if o is None or w["tg"] is None else w["tg"] - o)
                L = clean(leads)
                can = sum(w["tg"] is not None and w["tg"] > W for w in rows)
                fa = [onset(r["hist"]["steps"], r["hist"][metric], W, k) is not None for _, r in neg]
                res[f"{metric}|{W}|{k}"] = dict(onsets=ons, leads=leads, false_alarms=fa, at_start=first)
                R.p(f"    {W:>8} {k:>3} {len(clean(ons)):>5}/{len(runs):<3} {(L > 0).sum():>3}/{can:<3} "
                    f"{first:>5}/{len(runs):<3} {(f'{np.median(L):.0f}' if len(L) else 'n/a'):>12} "
                    f"{sum(fa):>7}/{len(neg)}")
    R.data["ipr_sensitivity"] = res


# ---------------------------------------------------------------- G/H. patching, logit PCA
def section_patching(R, run, device, sigma=0.3):
    R.h("G. Pre/post MLP activation patching sanity check (embedding noise)")
    model, data, labels, tr, te = restore(run, device)
    X, y = data[te], labels[te]
    torch.manual_seed(0)
    with torch.no_grad():
        _, cache = model.run_with_cache(X)
        pre, post = cache["blocks.0.mlp.hook_pre"], cache["blocks.0.mlp.hook_post"]
        emb = model.embed(X)
        noisy = emb + sigma * torch.randn_like(emb)
        corrupt = ("hook_embed", lambda a, hook: noisy)
        res = {}
        for name, extra in (("corrupted", []), ("pre patched", [("blocks.0.mlp.hook_pre", lambda a, hook: pre)]),
                            ("post patched", [("blocks.0.mlp.hook_post", lambda a, hook: post)])):
            logits = model.run_with_hooks(X, fwd_hooks=[corrupt] + extra)[:, -1, :]
            res[name] = float((logits.argmax(-1) == y).float().mean())
    R.p(f"  sigma={sigma}: " + ", ".join(f"{k} {100 * v:.2f}%" for k, v in res.items()))
    R.data["patching"] = res


def section_logit_pca(R, run, device, out):
    R.h("H. Logit geometry (Fig. 9a)")
    model, data, labels, _, _ = restore(run, device)
    with torch.no_grad():
        L = model(data)[:, -1, :P].double().cpu().numpy()
    y = labels.cpu().numpy()
    Lc = L - L.mean(1, keepdims=True)            # remove each input's mean logit (does not change the argmax)
    F_ = np.fft.fft(Lc, axis=1)                  # over the 97 classes
    pw = (np.abs(F_) ** 2).mean(0)
    fs = np.arange(1, (P - 1) // 2 + 1)
    pair = pw[fs] + pw[P - fs]
    order = np.argsort(pair)[::-1]
    fstar = int(fs[order[0]])
    phase = (fstar * y) % P                      # position on the frequency-f* circle
    Z = Lc - Lc.mean(0)
    U, S, _ = np.linalg.svd(Z, full_matrices=False)
    ev = S ** 2 / (S ** 2).sum()
    R.p(f"  seed {run['seed']}: top-5 logit frequencies {sorted(int(f) for f in fs[order[:5]])} carry "
        f"{100 * pair[order[:5]].sum() / pair.sum():.1f}% of the logit power (each input's mean removed)")
    R.p(f"  PCA of the logits: PC1-4 explain " + ", ".join(f"{100 * e:.1f}%" for e in ev[:4]))
    fig, axs = plt.subplots(1, 2, figsize=(9, 4.5))
    axs[0].scatter(U[:, 0] * S[0], U[:, 1] * S[1], c=phase, s=2, cmap="hsv")
    axs[0].set_title(f"logit PCA: PC1 {100 * ev[0]:.1f}%, PC2 {100 * ev[1]:.1f}%", fontsize=9)
    axs[1].scatter(F_[:, fstar].real, -F_[:, fstar].imag, c=phase, s=2, cmap="hsv")
    axs[1].set_title(f"logits projected on frequency {fstar} (cos, sin)", fontsize=9)
    axs[1].set_aspect("equal")
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"colour = {fstar}·(a+b) mod {P}", fontsize=9)
    savefig(out, "fig_logit_pca_transformer.png")


def section_freqs(R, runs, dc, device):
    R.h("J. Frequencies used (is d_c about 2 x the number of key frequencies?)")
    use = {"W_E": [], "W_U": []}
    for r in runs:
        wm = weight_matrices(restore(r, device)[0])
        use["W_E"].append(freq_usage(wm["W_E"].detach().float().cpu().numpy().T))      # DFT over tokens
        use["W_U"].append(freq_usage(wm["W_U"].detach().float().cpu().numpy()[:, :P]))  # DFT over classes
    for m, u in use.items():
        R.p(f"  {m}: top-5 share {fmt([x[0] for x in u], 2)};  #freq for 90% of power "
            f"{fmt([x[1] for x in u], 1)};  for 99% {fmt([x[2] for x in u], 1)}")
    R.p(f"  2 x #freq (90%, W_E) = {fmt([2 * x[1] for x in use['W_E']], 1)}")
    for m in ("W_E", "W_O", "W_in", "W_out", "W_U"):
        R.p(f"  d_c({m}) = {fmt(dc[m], 1, med=True)}")
    R.data["freq_usage"] = use


def section_dc_over_training(R, runs, device):
    R.h("K. Accuracy-preserving rank over training (does it change after grokking?)")
    R.p("  saved checkpoints from step 10,000 on; d_c is relative to that checkpoint's own")
    R.p("  test accuracy, so read it together with the accuracy column")
    mats = ("W_E", "W_in", "W_out", "W_U")
    rows = {}
    for r in runs:
        for s in sorted(r["ckpts"]):
            if s < 10000:
                continue
            model, data, labels, tr, te = restore(r, device, s)
            Xte, yte = data[te], labels[te]
            orig = acc(model, Xte, yte)
            d = rows.setdefault(s, {"acc": [], **{m: [] for m in mats}})
            d["acc"].append(100 * orig)
            for m in mats:
                ks, accs, _ = weight_curve(model, m, Xte, yte)
                d[m].append(first_k(ks, accs, orig, 0.99))
    R.p(f"  {'step':>7} {'test acc %':>22} " + " ".join(f"{'d_c(' + m + ')':>22}" for m in mats))
    for s in sorted(rows):
        d = rows[s]
        R.p(f"  {s:>7} {fmt(d['acc'], 1):>22} " + " ".join(f"{fmt(d[m], 1):>22}" for m in mats))
    R.data["dc_over_training"] = {str(s): v for s, v in rows.items()}


def margin_free_curve(model, name, data, labels):
    """Top-k truncations of one matrix judged by four criteria: 99% of test accuracy (as d_c),
    cross-entropy at most 1% above the clean value, and final-position logits (centred per
    input, since softmax ignores a constant shift) within 10% / 5% of the clean logits.
    The last three do not depend on the margin. Also returns the median logit margin."""
    ce = torch.nn.functional.cross_entropy
    with torch.no_grad():
        base = model(data)[:, -1, :]
    base_c = base - base.mean(-1, keepdim=True)
    acc0 = float((base.argmax(-1) == labels).float().mean())
    ce0 = float(ce(base, labels))
    rest = base.clone()
    rest.scatter_(1, labels[:, None], -float("inf"))
    margin = float((base.gather(1, labels[:, None]).squeeze(1) - rest.max(-1).values).median())
    W0 = weight_matrices(model)[name].detach().clone()
    n = int(torch.linalg.svdvals(as2d(W0).float()).shape[0])
    ks = sorted(set(range(1, min(21, n + 1))) | set(range(25, n + 1, 5)) | {n})
    d = dict(d_acc=None, d_ce=None, d_out10=None, d_out05=None)
    for k in ks:
        Wk, _ = truncated(W0, range(k))
        set_weight(model, name, Wk)
        with torch.no_grad():
            lg = model(data)[:, -1, :]
        rel = float(((lg - lg.mean(-1, keepdim=True)) - base_c).norm() / base_c.norm())
        for key, ok in (("d_acc", float((lg.argmax(-1) == labels).float().mean()) >= 0.99 * acc0),
                        ("d_ce", float(ce(lg, labels)) <= 1.01 * ce0),
                        ("d_out10", rel <= 0.10), ("d_out05", rel <= 0.05)):
            if d[key] is None and ok:
                d[key] = k
        if all(v is not None for v in d.values()):
            break
    set_weight(model, name, W0)
    return d, 100 * acc0, margin


def section_margin_free(R, runs, device):
    R.h("L. Is d_c a margin effect? Margin-free criteria, each seed aligned at its own t_g")
    R.p("  d by criterion: 99% of test accuracy (d_c, margin-dependent); cross-entropy <= 1.01 x clean;")
    R.p("  centred final-position logits within 10% / 5% of the clean logits (relative Frobenius norm)")
    R.p("  checkpoints per seed: the last one before t_g, the first one at or after t_g, and the final one")
    mats = ("W_E", "W_in", "W_out", "W_U")
    crit = (("d_acc", "d (99% acc)"), ("d_ce", "d (CE +1%)"), ("d_out10", "d (logits 10%)"),
            ("d_out05", "d (logits 5%)"))
    out = {}
    for lab in ("last checkpoint before t_g", "first checkpoint at or after t_g", "final checkpoint"):
        res = {"acc": [], "margin": [], **{f"{m}|{c}": [] for m in mats for c, _ in crit}}
        for r in runs:
            tg = t_grok(r["hist"])
            steps = sorted(r["ckpts"])
            if lab.startswith("last"):
                cand = [s for s in steps if s < tg]
                if not cand:
                    continue
                s = cand[-1]
            elif lab.startswith("first"):
                s = min(s for s in steps if s >= tg)
            else:
                s = r["final_step"]
            model, data, labels, tr, te = restore(r, device, s)
            first = True
            for m in mats:
                d, acc0, margin = margin_free_curve(model, m, data[te], labels[te])
                if first:
                    res["acc"].append(acc0)
                    res["margin"].append(margin)
                    first = False
                for c, _ in crit:
                    res[f"{m}|{c}"].append(d[c])
        out[lab] = res
        R.p(f"\n  [{lab}]  n={len(res['acc'])}  test acc {fmt(res['acc'], 1)} %  "
            f"median logit margin {fmt(res['margin'], 2)}")
        R.p(f"  {'matrix':<8}" + "".join(f"{h:>24}" for _, h in crit))
        for m in mats:
            R.p(f"  {m:<8}" + "".join(f"{fmt(res[f'{m}|{c}'], 1):>24}" for c, _ in crit))
    R.data["margin_free"] = out


def section_negatives(R, neg, device):
    R.h("N. Runs that did not grok within the step budget (negative controls)")
    if not neg:
        R.p("  none")
        return
    R.p("  k99 = number of singular directions for 99% of the variance (as in section A)")
    R.p(f"  {'run':<18} {'train':>7} {'test':>7}  " + " ".join(f"{m:>6}" for m in MATS))
    out = {}
    for lab, r in neg:
        wm = weight_matrices(restore(r, device)[0])
        k99 = {}
        for m in MATS:
            s = torch.linalg.svdvals(as2d(wm[m]).detach().float()).cpu().numpy()
            k99[m] = k_at(np.cumsum(s ** 2) / (s ** 2).sum(), .99)
        h = r["hist"]
        R.p(f"  {lab:<18} {100 * h['train_acc'][-1]:>6.1f}% {100 * h['test_acc'][-1]:>6.1f}%  "
            + " ".join(f"{k99[m]:>6}" for m in MATS))
        out[lab] = dict(train=float(h["train_acc"][-1]), test=float(h["test_acc"][-1]), k99=k99)
    R.data["negatives"] = out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results")
    ap.add_argument("--out", default=None)
    ap.add_argument("--wd", type=float, default=1.0, help="weight decay of the main runs")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    out = args.out or os.path.join(args.runs, "analysis_transformer")
    os.makedirs(out, exist_ok=True)
    runs_all = load_runs(args.runs, args.wd)
    if not runs_all:
        raise SystemExit(f"no runs in {args.runs}/transformer/wd{args.wd:g}")
    runs = [r for r in runs_all if grokked(r)]
    neg = [(f"wd={args.wd:g} seed {r['seed']}", r) for r in runs_all if not grokked(r)]
    for wd in all_wds(args.runs):
        if wd != args.wd:
            neg += [(f"wd={wd:g} seed {r['seed']}", r) for r in load_runs(args.runs, wd) if not grokked(r)]
    if not runs:
        raise SystemExit(f"none of the wd={args.wd:g} runs reached 95% test accuracy")
    R = Report()
    R.p(f"{len(runs_all)} transformer runs at wd={args.wd:g}, config {runs_all[0]['config']}")
    R.p(f"{len(runs)} grokked (final test accuracy >= 95%) and are used in sections A-H and J;")
    R.p(f"non-grokked runs (negative controls, section N): {[lab for lab, _ in neg]}")
    section_spectra(R, runs, args.device, out)
    dc = section_causal_dims(R, runs, args.device)
    section_alignment(R, runs, args.device)
    section_timing(R, runs, neg, out)
    section_patching(R, runs[0], args.device)
    section_logit_pca(R, runs[0], args.device, out)
    section_alignment_vs_wd(R, args.runs, args.device)
    section_freqs(R, runs, dc, args.device)
    section_dc_over_training(R, runs, args.device)
    section_margin_free(R, runs, args.device)
    section_negatives(R, neg, args.device)
    R.save(out)
    print(f"\nwrote {out}/summary.txt, summary.json and figures")


if __name__ == "__main__":
    main()
