"""
analyze_mlp.py -- every MLP table and figure, recomputed from saved runs.

  python analyze_mlp.py --runs results --out results/analysis_mlp

Reads <runs>/mlp/<exp>/seed*.pt (written by run_mlp.py) and writes
  summary.txt   human-readable tables
  summary.json  
  fig_*.png     figures
Sections run only if their experiments exist, so it can be re-run as
results come in.
"""
import argparse
import glob
import json
import os
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats
from scipy.signal import correlate

from common_mlp import (P, accuracy, compression_onset, detect_onset, effective_rank, final_model,
                        grokking_onset, load_run, model_from_weights, run_data)

try:
    from statsmodels.tsa.stattools import adfuller, grangercausalitytests
    HAVE_SM = True
except ImportError:
    HAVE_SM = False

THRESHOLDS = (0.90, 0.95, 0.99, 0.999)


# ================================================================ utilities
class Report:
    def __init__(self):
        self.lines, self.data = [], {}

    def h(self, title):
        self.lines += ["", "=" * 76, title, "=" * 76]
        print(f"\n== {title}", flush=True)

    def p(self, *a):
        self.lines.append(" ".join(str(x) for x in a))

    def save(self, out):
        with open(os.path.join(out, "summary.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(self.lines) + "\n")
        with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=1, default=_json)


def _json(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def clean(xs):
    return np.asarray([x for x in xs if x is not None], dtype=float)


def mstd(xs, ddof=1):
    x = clean(xs)
    if len(x) == 0:
        return float("nan"), float("nan"), 0
    sd = x.std(ddof=ddof) if len(x) > 1 else 0.0
    return float(x.mean()), float(sd), len(x)


def fmt(xs, nd=1, ddof=1):
    m, s, n = mstd(xs, ddof)
    return f"{m:.{nd}f} ± {s:.{nd}f} (n={n})" if n else "n/a"


def load_exp(root, name):
    paths = sorted(glob.glob(os.path.join(root, "mlp", name, "seed*.pt")),
                   key=lambda p: int(os.path.basename(p)[4:-3]))
    return [load_run(p) for p in paths]


def subspace_cos(A, B):
    """Mean cosine of the principal angles between span(A) and span(B)
    (A, B with orthonormal columns)."""
    return float(np.linalg.svd(A.T @ B, compute_uv=False).mean())


def orth(M):
    return np.linalg.qr(M)[0]


def random_baseline(k, d, n=300, seed=0):
    rng = np.random.default_rng(seed)
    v = [subspace_cos(orth(rng.standard_normal((d, k))), orth(rng.standard_normal((d, k))))
         for _ in range(n)]
    return float(np.mean(v)), float(np.std(v))


def acc_with_W1(model, W1, X, y):
    with torch.no_grad():
        h = (W1 @ X.T).T / np.sqrt(X.shape[1])
        return accuracy(model.readout(model.act(h)), y)


def savefig(out, name):
    plt.tight_layout()
    plt.savefig(os.path.join(out, name), dpi=200)
    plt.close()


# ================================================================ A. onsets
def share_after_tg(steps, er, tg):
    """Share of the decrease in ER, from its maximum up to t_g down to its
    final value, that happens after t_g (1 = all of it, 0 = none of it).
    Same definition in analyze_transformer.py."""
    if tg is None:
        return None
    steps, er = np.asarray(steps), np.asarray(er, dtype=float)
    peak, at, end = er[steps <= tg].max(), er[steps == tg][0], er[-1]
    return float((at - end) / (peak - end)) if peak > end else None


def steepest_drop(steps, er, start, smooth=5):
    """Step of the steepest decrease of ER (moving average over `smooth` logged
    points), ignoring steps before `start` and the last 4 logged steps.
    Same definition in analyze_transformer.py (there with start=4000)."""
    st, er = np.asarray(steps), np.asarray(er, dtype=float)
    d = np.gradient(np.convolve(er, np.ones(smooth) / smooth, mode="same"), st)
    ok = (st >= start) & (st <= st[-1] - 4 * (st[1] - st[0]))
    return int(st[ok][np.argmin(d[ok])])


def onset_rows(runs):
    rows = []
    for r in runs:
        h = r["hist"]
        st = h["steps"]
        tg = grokking_onset(st, h["test_acc"])
        tc = compression_onset(st, h["W1_erank"])
        ti = detect_onset(st, h["fourier_ipr"], 5000, 3, "up")
        ts = steepest_drop(st, h["W1_erank"], 1000)
        rows.append(dict(seed=r["seed"], tg=tg, tc=tc, tipr=ti,
                         lag=None if tg is None or tc is None else tc - tg,
                         lead=None if tg is None or ti is None else tg - ti,
                         after=share_after_tg(st, h["W1_erank"], tg),
                         steep_lag=None if tg is None else ts - tg))
    return rows


def section_onsets(R, runs):
    R.h("A. Onset times (Table 3) -- main runs")
    rows = onset_rows(runs)
    R.p(f"{'seed':>4} {'t_g':>8} {'t_c':>8} {'lag':>8} {'t_IPR':>8} {'IPR lead':>9}")
    for w in rows:
        R.p(f"{w['seed']:>4} {str(w['tg']):>8} {str(w['tc']):>8} {str(w['lag']):>8} "
            f"{str(w['tipr']):>8} {str(w['lead']):>9}")
    for key, lab in [("tg", "t_g (test acc >= 95%)"), ("tc", "t_c (ER < 99% of running max)"),
                     ("lag", "lag t_c - t_g"), ("tipr", "t_IPR (window 0-5000, 3 sd)"),
                     ("lead", "IPR lead t_g - t_IPR")]:
        vals = [w[key] for w in rows]
        R.p(f"  {lab:<32} {fmt(vals, 0)}   [ddof=0 sd: {mstd(vals, 0)[1]:.0f}]")
    lags = clean([w["lag"] for w in rows])
    if len(lags) > 1:
        t, pv = stats.ttest_1samp(lags, 0, alternative="greater")
        R.p(f"  seeds with t_c > t_g: {(lags > 0).sum()}/{len(lags)}   one-sided t-test: "
            f"t={t:.3f}, p={pv:.2e}, standardized effect={lags.mean() / lags.std(ddof=1):.2f}")
    steps_total = runs[0]["final_step"] + 1
    m, _, _ = mstd([w["lag"] for w in rows])
    R.p(f"  mean lag as % of training: {100 * m / steps_total:.1f}%;  as % of t_g: "
        f"{fmt([None if w['lag'] is None else 100 * w['lag'] / w['tg'] for w in rows], 1)}")
    R.p(f"  share of the ER(W1) decrease that happens after t_g: "
        f"{fmt([w['after'] for w in rows], 2)}  (compare analyze_transformer.py section E)")
    sl = clean([w["steep_lag"] for w in rows])
    if len(sl) > 1:
        t, pv = stats.ttest_1samp(sl, 0, alternative="greater")
        R.p(f"  steepest ER(W1) decrease (smoothed, from step 1,000) minus t_g: {fmt(sl, 0)}, median "
            f"{np.median(sl):.0f};  after t_g in {(sl > 0).sum()}/{len(sl)};  one-sided t={t:.2f}, p={pv:.1e}"
            f"  (same measure in analyze_transformer.py section E)")
    R.p("\n  accuracy and effective rank before t_g (Stages 1 and 2)")
    R.p(f"  {'step':>7} {'train acc %':>22} {'test acc %':>22} {'ER(W1)':>22}")
    pre = {}
    for s0 in (1000, 4000, 6000, 7000, 8000, 9000, 10000, 11000, 12000):
        v = {k: [] for k in ("train", "test", "er")}
        for r in runs:
            st = list(r["hist"]["steps"])
            if s0 in st:
                i = st.index(s0)
                v["train"].append(100 * r["hist"]["train_acc"][i])
                v["test"].append(100 * r["hist"]["test_acc"][i])
                v["er"].append(r["hist"]["W1_erank"][i])
        pre[s0] = v
        R.p(f"  {s0:>7} {fmt(v['train'], 2):>22} {fmt(v['test'], 2):>22} {fmt(v['er'], 2):>22}")
    for thr in (0.90, 0.99, 0.999):
        first = [next((int(s0) for s0, a in zip(r["hist"]["steps"], r["hist"]["train_acc"]) if a >= thr), None)
                 for r in runs]
        R.p(f"  first step with train acc >= {100 * thr:g}%: {fmt(first, 0)}")
    R.data["onsets"] = rows
    R.data["before_tg"] = {str(k): v for k, v in pre.items()}
    return rows


# ================================================================ B. xcorr
def xcorr(h, log_every):
    de, da = np.gradient(h["W1_erank"]), np.gradient(h["test_acc"])
    de = (de - de.mean()) / (de.std() + 1e-10)
    da = (da - da.mean()) / (da.std() + 1e-10)
    xc = correlate(de, da, mode="full") / len(de)
    lags = np.arange(-(len(de) - 1), len(de)) * log_every
    return lags, xc


def section_xcorr(R, runs, out):
    R.h("B. Cross-correlation of dER/dt and dAcc/dt (Fig. 7b)")
    le = runs[0]["config"]["log_every"]
    curves = [xcorr(r["hist"], le) for r in runs]
    n = min(len(c[1]) for c in curves)
    lags = curves[0][0][:n]
    X = np.stack([c[1][:n] for c in curves])
    mean, sd = X.mean(0), X.std(0)
    trough = int(lags[np.argmin(mean)])
    per_seed = [int(lags[np.argmin(x)]) for x in X]
    neg = (lags >= -80000) & (lags <= -5000)
    R.p(f"  trough of mean curve at tau = {trough:+d} steps")
    R.p(f"  per-seed trough lags: {fmt(per_seed, 0)}")
    if neg.any():
        R.p(f"  max |C(tau)| for tau in [-80000, -5000]: {np.abs(mean[neg]).max():.3f}")
    R.data["xcorr"] = dict(trough=trough, per_seed=per_seed)
    plt.figure(figsize=(8, 3.5))
    plt.plot(lags, mean, lw=1.5, label=f"mean ({len(runs)} seeds)")
    plt.fill_between(lags, mean - sd, mean + sd, alpha=0.2)
    plt.axvline(trough, color="crimson", ls="--", lw=1, label=f"trough {trough:+d}")
    plt.axvline(0, color="k", lw=0.5)
    plt.xlabel("lag tau (steps; tau > 0: rank changes lag accuracy changes)")
    plt.ylabel("C(tau)")
    plt.legend()
    savefig(out, "fig_leadlag_xcorr.png")


# ================================================================ C. Granger
def _ssr(y, X):
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    r = y - X @ beta
    return float(r @ r)


def _granger(cause, effect, maxlag):
    """F, p, lag for H0: `cause` does not help predict `effect`.
    The ssr F-test of statsmodels' grangercausalitytests, written out in numpy
    so it does not depend on the statsmodels version; lag chosen as in NEW.py
    (min AIC of the restricted model, each lag on its own trimmed sample)."""
    cause, effect = np.asarray(cause, float), np.asarray(effect, float)
    N, best = len(effect), None
    for L in range(1, maxlag + 1):
        y = effect[L:]
        n = len(y)
        own = np.column_stack([effect[L - j:N - j] for j in range(1, L + 1)] + [np.ones(n)])
        joint = np.column_stack([own] + [cause[L - j:N - j] for j in range(1, L + 1)])
        ssr_r, ssr_u = _ssr(y, own), _ssr(y, joint)
        df = n - joint.shape[1]
        F = (ssr_r - ssr_u) / ssr_u / L * df
        aic = n * (np.log(2 * np.pi) + np.log(ssr_r / n) + 1) + 2 * own.shape[1]
        if best is None or aic < best[0]:
            best = (aic, float(F), float(stats.f.sf(F, L, df)), L)
    return best[1:]


def _granger_statsmodels(cause, effect, maxlag):
    """NEW.py's original call, used only to cross-check _granger."""
    gc = grangercausalitytests(np.column_stack([effect, cause]), maxlag=maxlag, verbose=False)
    aic = {L: gc[L][1][0].aic for L in range(1, maxlag + 1)}
    L = min(aic, key=aic.get)
    F, pv = gc[L][0]["ssr_ftest"][:2]
    return float(F), float(pv), L


def _adf(x):
    if not HAVE_SM:
        return float("nan")
    try:
        return float(adfuller(x, autolag="AIC")[1])
    except Exception:
        return float("nan")


def _make_stationary(x, max_extra=2):
    n = 0
    while _adf(x) > 0.05 and n < max_extra:
        x, n = np.diff(x), n + 1
    return x, n


def section_granger(R, runs, maxlag=4):
    R.h("C. Granger-type test of predictive precedence (App. / Sec. 6.2)")
    if not HAVE_SM:
        R.p("  statsmodels not installed: F-tests run, ADF stationarity checks reported as n/a")
    out = {"original": [], "stationary": []}
    for r in runs:
        h = r["hist"]
        de, da = np.diff(h["W1_erank"]), np.diff(h["test_acc"])
        row = dict(seed=r["seed"], adf_dER=_adf(de), adf_dAcc=_adf(da))
        try:
            row["C->G"] = _granger(de, da, maxlag)
            row["G->C"] = _granger(da, de, maxlag)
        except Exception as e:
            row["error"] = str(e)
        out["original"].append(row)
        # stationarity-checked version: difference until ADF p < 0.05 (<= 2 more times)
        de2, n1 = _make_stationary(de)
        da2, n2 = _make_stationary(da)
        m = min(len(de2), len(da2))
        de2, da2 = de2[-m:], da2[-m:]
        row2 = dict(seed=r["seed"], extra_diffs=(n1, n2), adf_dER=_adf(de2), adf_dAcc=_adf(da2))
        try:
            row2["C->G"] = _granger(de2, da2, maxlag)
            row2["G->C"] = _granger(da2, de2, maxlag)
        except Exception as e:
            row2["error"] = str(e)
        out["stationary"].append(row2)

    # cross-check the numpy F-test against statsmodels (NEW.py's call) on every seed
    if HAVE_SM:
        diffs, errs = [], []
        for r in runs:
            de, da = np.diff(r["hist"]["W1_erank"]), np.diff(r["hist"]["test_acc"])
            try:
                a, b = _granger(de, da, maxlag), _granger_statsmodels(de, da, maxlag)
                diffs.append(abs(a[0] - b[0]) / max(abs(b[0]), 1e-12))
            except Exception as e:
                errs.append(f"{type(e).__name__}: {e}")
        if diffs:
            R.p(f"  check vs statsmodels: max relative difference in F = {max(diffs):.1e} "
                f"over {len(diffs)} seeds")
        if errs:
            R.p(f"  statsmodels' grangercausalitytests failed in {len(errs)} seeds ({errs[0]}); "
                f"the numpy F-test below does not use it")

    for key, title in [("original", "first differences only (as in the paper)"),
                       ("stationary", "differenced until ADF p < 0.05")]:
        rows = [w for w in out[key] if "C->G" in w]
        R.p(f"\n  [{title}]  n={len(rows)}")
        failed = [w for w in out[key] if "error" in w]
        if failed:
            R.p(f"    failed in {len(failed)} seeds, e.g. seed {failed[0]['seed']}: {failed[0]['error']}")
        if not rows:
            continue
        R.p(f"    ADF p (dER): {fmt([w['adf_dER'] for w in rows], 3)};  "
            f"stationary in {sum(w['adf_dER'] < 0.05 for w in rows)}/{len(rows)} seeds")
        R.p(f"    ADF p (dAcc): {fmt([w['adf_dAcc'] for w in rows], 3)};  "
            f"stationary in {sum(w['adf_dAcc'] < 0.05 for w in rows)}/{len(rows)} seeds")
        for d, lab in [("C->G", "compression -> grokking"), ("G->C", "grokking -> compression")]:
            F = [w[d][0] for w in rows]
            pv = [w[d][1] for w in rows]
            R.p(f"    H0: {lab:<26} mean F={np.mean(F):.2f}  mean p={np.mean(pv):.3f}  "
                f"median p={np.median(pv):.3f}  rejected in {sum(p < 0.05 for p in pv)}/{len(pv)}")
    R.data["granger"] = out


# ================================================================ D. onset sensitivity
ONSET_METRICS = [("fourier_ipr", "up"), ("spectral_ipr", "up"), ("W1_erank", "down"),
                 ("W1_srank", "down"), ("W1_l0", "both"), ("W1_hoyer", "both"),
                 ("grad_norm", "both")]


def section_onset_sensitivity(R, runs, neg_runs):
    R.h("D. Label-free onset detection: windows x thresholds, all metrics (Sec. 6.1, Table 4)")
    R.p("  detected = seeds where the detector fires; early = fires before t_g;")
    R.p("  at start = fires at the first logged step after the window, i.e. as soon as it")
    R.p("  is allowed to (a trend, not an event); false alarms = random-label runs")
    R.p("  (never generalize) where it fires.")
    tgs = {r["seed"]: grokking_onset(r["hist"]["steps"], r["hist"]["test_acc"]) for r in runs}
    res = {}
    for metric, d in ONSET_METRICS:
        R.p(f"\n  metric={metric} ({d})")
        R.p(f"    {'window':>8} {'k':>3} {'detected':>9} {'early':>7} {'at start':>9} "
            f"{'median lead':>12} {'false alarms':>13}")
        for W in (2500, 5000, 10000):
            for k in (2, 3, 4):
                ons, leads, first = [], [], 0
                for r in runs:
                    st = np.asarray(r["hist"]["steps"])
                    o = detect_onset(st, r["hist"][metric], W, k, d)
                    ons.append(o)
                    elig = st[st >= W]
                    first += int(o is not None and len(elig) > 0 and o == int(elig[0]))
                    tg = tgs[r["seed"]]
                    leads.append(None if o is None or tg is None else tg - o)
                L = clean(leads)
                fa = [detect_onset(r["hist"]["steps"], r["hist"][metric], W, k, d) is not None
                      for r in neg_runs]
                key = f"{metric}|{W}|{k}"
                res[key] = dict(onsets=ons, leads=leads, false_alarms=fa, at_start=first)
                med = f"{np.median(L):.0f}" if len(L) else "n/a"
                R.p(f"    {W:>8} {k:>3} {len(clean(ons)):>5}/{len(runs):<3} {(L > 0).sum():>3}/{len(runs):<3} "
                    f"{first:>5}/{len(runs):<3} {med:>12} {sum(fa):>7}/{len(neg_runs)}")
    R.data["onset_sensitivity"] = res


# ================================================================ E. causal dims
def topk_curve(model, X, y):
    W1 = model.W1_full.detach()
    U, S, Vh = torch.linalg.svd(W1, full_matrices=False)
    orig = acc_with_W1(model, W1, X, y)
    ks = np.arange(1, len(S) + 1)
    accs = np.array([acc_with_W1(model, (U[:, :k] * S[:k]) @ Vh[:k], X, y) for k in ks])
    return ks, accs, orig, (U, S, Vh)


def d_at(ks, accs, orig, tau):
    ok = np.where(accs >= tau * orig)[0]
    return int(ks[ok[0]]) if len(ok) else None


def section_causal_dims(R, runs, out):
    R.h("E. Causal dimensionality of W1 (App. B, Fig. 3a), controls (Table 2), complement tests")
    curves, dims = [], {t: [] for t in THRESHOLDS}
    kvar = {0.95: [], 0.99: []}
    ctrl = {"top": [], "bottom": [], "rand_subset": [], "rand_subspace": []}
    comp = {"keep_C": [], "keep_Cperp": []}
    noise_scales = (0.1, 0.3, 1.0, 3.0)
    noise = {(sub, s): [] for sub in ("C", "Cperp") for s in noise_scales}
    for r in runs:
        X, y, _, te = run_data(r)
        model = final_model(r)
        Xte, yte = X[te], y[te]
        ks, accs, orig, (U, S, Vh) = topk_curve(model, Xte, yte)
        curves.append(accs)
        cum = torch.cumsum(S ** 2, 0) / (S ** 2).sum()
        for q in kvar:
            kvar[q].append(int((cum < q).sum()) + 1)
        for t in THRESHOLDS:
            dims[t].append(d_at(ks, accs, orig, t))
        k = dims[0.99][-1]
        if k is None:
            continue
        rng = np.random.default_rng(r["seed"])
        n = len(S)
        rec = lambda idx: (U[:, idx] * S[idx]) @ Vh[idx]
        ctrl["top"].append(acc_with_W1(model, rec(list(range(k))), Xte, yte))
        ctrl["bottom"].append(acc_with_W1(model, rec(list(range(n - k, n))), Xte, yte))
        ctrl["rand_subset"].append(np.mean([
            acc_with_W1(model, rec([int(i) for i in sorted(rng.choice(n, k, replace=False))]), Xte, yte)
            for _ in range(10)]))
        W1 = model.W1_full.detach()
        sub = []
        for _ in range(10):
            Q = torch.tensor(orth(rng.standard_normal((W1.shape[0], k))), dtype=W1.dtype)
            sub.append(acc_with_W1(model, Q @ (Q.T @ W1), Xte, yte))
        ctrl["rand_subspace"].append(np.mean(sub))
        # complement interventions on the pre-activations h (hidden space)
        with torch.no_grad():
            h = model.pre(Xte)
            Pc = U[:, :k] @ U[:, :k].T
            I = torch.eye(Pc.shape[0])
            comp["keep_C"].append(accuracy(model.readout(model.act(h @ Pc)), yte))
            comp["keep_Cperp"].append(accuracy(model.readout(model.act(h @ (I - Pc))), yte))
            hn = h.norm(dim=1, keepdim=True)
            gen = torch.Generator().manual_seed(r["seed"])
            for subn, Pm in (("C", Pc), ("Cperp", I - Pc)):
                for s in noise_scales:
                    a = []
                    for _ in range(5):
                        g = torch.randn(h.shape, generator=gen) @ Pm
                        g = g / (g.norm(dim=1, keepdim=True) + 1e-12)
                        a.append(accuracy(model.readout(model.act(h + s * hn * g)), yte))
                    noise[(subn, s)].append(float(np.mean(a)))

    R.p("  App. B -- d_tau = smallest k whose top-k truncation keeps >= tau x clean accuracy")
    for t in THRESHOLDS:
        R.p(f"    tau={t:<6} d = {fmt(dims[t], 1)}")
    R.p(f"  variance counts of W1 (directions carrying 95% / 99% of sum sigma^2): "
        f"k95 = {fmt(kvar[0.95], 1)}, k99 = {fmt(kvar[0.99], 1)}")
    R.p("\n  Table 2 (MLP row) -- truncations of W1 at k = d_0.99 (per seed), test accuracy")
    for key, lab in [("top", "top-k"), ("bottom", "bottom-k"),
                     ("rand_subset", "random k of the singular directions (10 draws)"),
                     ("rand_subspace", "random k-dim subspace of R^128 (10 draws)")]:
        R.p(f"    {lab:<48} {fmt([100 * v for v in ctrl[key]], 2)} %")
    R.p("\n  Complement interventions on pre-activations h (C = top-d_0.99 left singular subspace)")
    R.p(f"    keep only C      : {fmt([100 * v for v in comp['keep_C']], 2)} %")
    R.p(f"    keep only C-perp : {fmt([100 * v for v in comp['keep_Cperp']], 2)} %")
    R.p("    noise of norm s*||h|| added inside C or inside C-perp:")
    for s in noise_scales:
        R.p(f"      s={s:<4}  in C: {fmt([100 * v for v in noise[('C', s)]], 2)} %   "
            f"in C-perp: {fmt([100 * v for v in noise[('Cperp', s)]], 2)} %")
    R.data["causal_dims"] = {str(t): dims[t] for t in THRESHOLDS}
    R.data["variance_counts"] = {str(q): v for q, v in kvar.items()}
    R.data["controls"] = ctrl
    R.data["complement"] = dict(comp, noise={f"{a}|{b}": v for (a, b), v in noise.items()})

    C = np.stack(curves)
    ks = np.arange(1, C.shape[1] + 1)
    plt.figure(figsize=(6, 4))
    plt.plot(ks, C[0], lw=2, label=f"seed {runs[0]['seed']}")
    plt.fill_between(ks, C.mean(0) - C.std(0), C.mean(0) + C.std(0), alpha=0.25,
                     label=f"mean ± sd ({len(runs)} seeds)")
    d99 = clean(dims[0.99])
    if len(d99):
        plt.axvline(d99.mean(), color="green", ls=":", label=f"mean d_0.99 = {d99.mean():.1f}")
    plt.xlabel("rank k of truncated W1")
    plt.ylabel("test accuracy")
    plt.legend()
    savefig(out, "fig_causal_dimension.png")
    return dims


def section_dc_over_training(R, runs, out):
    R.h("E2. Causal dimensionality over training (does the sufficient subspace change in Stage 4?)")
    R.p("  every saved checkpoint from step 10,000 on; d_tau is relative to that checkpoint's own")
    R.p("  test accuracy, so read it together with the accuracy column")
    rows = {}
    for r in runs:
        X, y, _, te = run_data(r)
        for s, w in sorted(r["ckpts"].items()):
            if s < 10000:
                continue
            m = model_from_weights(w["W1"], w["W2"], r["config"]["act"])
            ks, accs, orig, _ = topk_curve(m, X[te], y[te])
            d = rows.setdefault(s, dict(acc=[], d90=[], d99=[], er=[], nfreq=[]))
            d["acc"].append(100 * orig)
            d["d90"].append(d_at(ks, accs, orig, 0.90))
            d["d99"].append(d_at(ks, accs, orig, 0.99))
            d["er"].append(effective_rank(w["W1"]))
            d["nfreq"].append(freq_usage(w["W1"].numpy()[:, :P])[1])
    R.p(f"  {'step':>7} {'test acc %':>20} {'d_0.9':>20} {'d_0.99':>20} {'ER(W1)':>20} {'#freq (90%)':>20}")
    for s in sorted(rows):
        d = rows[s]
        R.p(f"  {s:>7} {fmt(d['acc'], 1):>20} {fmt(d['d90'], 1):>20} {fmt(d['d99'], 1):>20} "
            f"{fmt(d['er'], 1):>20} {fmt(d['nfreq'], 1):>20}")
    R.data["dc_over_training"] = {str(s): v for s, v in rows.items()}
    st = sorted(rows)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for key, lab in (("d99", "$d_{0.99}$"), ("d90", "$d_{0.9}$")):
        mu = [mstd(rows[s][key])[0] for s in st]
        sd = [mstd(rows[s][key])[1] for s in st]
        ax.errorbar(st, mu, yerr=sd, marker="o", capsize=3, label=lab)
    ax.set_xscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("directions of $W_1$ needed")
    ax2 = ax.twinx()
    ax2.plot(st, [mstd(rows[s]["er"])[0] for s in st], "k--", label="ER$(W_1)$")
    ax2.set_ylabel("effective rank")
    h1, l1 = ax.get_legend_handles_labels()     # one legend, in the empty upper-right corner
    h2, l2 = ax2.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)
    savefig(out, "fig_dc_over_training.png")


# ================================================================ E4. margin-free d
def truncation_criteria(model, W1, X, y):
    """Top-k truncations of W1 judged by four criteria: 99% of the test accuracy (as d_0.99),
    test MSE at most 1% above the clean value, and outputs within 10% / 5% of the clean
    outputs (relative Frobenius norm). The last three do not depend on the margin."""
    U, S, Vh = torch.linalg.svd(W1, full_matrices=False)
    Y = torch.nn.functional.one_hot(y, P).float()
    f0 = model.readout(model.act((W1 @ X.T).T / np.sqrt(X.shape[1])))
    acc0, mse0, n0 = accuracy(f0, y), float(((f0 - Y) ** 2).mean()), float(f0.norm())
    fc = f0.gather(1, y[:, None]).squeeze(1)
    rest = f0.clone()
    rest.scatter_(1, y[:, None], -float("inf"))
    margin = float((fc - rest.max(1).values).median())
    d = dict(d_acc=None, d_mse=None, d_out10=None, d_out05=None)
    for k in range(1, len(S) + 1):
        fk = model.readout(model.act((((U[:, :k] * S[:k]) @ Vh[:k]) @ X.T).T / np.sqrt(X.shape[1])))
        rel = float((fk - f0).norm()) / n0
        for key, ok in (("d_acc", accuracy(fk, y) >= 0.99 * acc0),
                        ("d_mse", float(((fk - Y) ** 2).mean()) <= 1.01 * mse0),
                        ("d_out10", rel <= 0.10), ("d_out05", rel <= 0.05)):
            if d[key] is None and ok:
                d[key] = k
        if all(v is not None for v in d.values()):
            break
    return d, 100 * acc0, margin


def section_margin_free(R, runs):
    R.h("E4. Is the narrowing of d_0.99 a margin effect? Margin-free criteria over training")
    R.p("  d by criterion: 99% of test accuracy (d_0.99, margin-dependent); test MSE <= 1.01 x clean;")
    R.p("  outputs within 10% / 5% of the clean outputs (relative Frobenius norm); test set, main runs")
    cols = [("test_acc", "test acc %", 1), ("margin", "median margin", 3), ("d_acc", "d (99% acc)", 1),
            ("d_mse", "d (MSE +1%)", 1), ("d_out10", "d (outputs 10%)", 1), ("d_out05", "d (outputs 5%)", 1)]
    rows = {}
    for r in runs:
        X, y, _, te = run_data(r)
        for s, w in sorted(r["ckpts"].items()):
            if s < 10000:
                continue
            m = model_from_weights(w["W1"], w["W2"], r["config"]["act"])
            d, acc, margin = truncation_criteria(m, w["W1"].float(), X[te], y[te])
            row = rows.setdefault(s, {k: [] for k, _, _ in cols})
            row["test_acc"].append(acc)
            row["margin"].append(margin)
            for k, v in d.items():
                row[k].append(v)
    R.p("  " + f"{'step':>7}" + "".join(f"{lab:>24}" for _, lab, _ in cols))
    for s in sorted(rows):
        R.p("  " + f"{s:>7}" + "".join(f"{fmt(rows[s][k], nd):>24}" for k, _, nd in cols))
    R.data["margin_free"] = {str(s): v for s, v in rows.items()}


# ================================================================ E3. Stage-4 widening
def _margins(f, y):
    """Correct-class output and its margin over the largest other output."""
    fc = f.gather(1, y[:, None]).squeeze(1)
    rest = f.clone()
    rest.scatter_(1, y[:, None], -float("inf"))
    return fc - rest.max(1).values, fc


def d99_at(r, s):
    X, y, _, te = run_data(r)
    w = r["ckpts"][s]
    ks, accs, orig, _ = topk_curve(model_from_weights(w["W1"], w["W2"], r["config"]["act"]), X[te], y[te])
    return d_at(ks, accs, orig, 0.99)


def widening_stats(r, s, K):
    """Checkpoint s: weight growth, and how much removing the directions of W1
    beyond its top K changes the output. With z = h^2 and h = h_top + h_rest,
    z = h_top^2 + 2 h_top h_rest + h_rest^2, so the effect of h_rest on the
    output grows with the size of h_top (the cross term)."""
    X, y, _, te = run_data(r)
    Xte, yte = X[te], y[te]
    w = r["ckpts"][s]
    m = model_from_weights(w["W1"], w["W2"], r["config"]["act"])
    ks, accs, orig, (U, S, Vh) = topk_curve(m, Xte, yte)
    margin, fc = _margins(m(Xte), yte)
    # gradient-flow forces on ||theta||^2 (MSE, quadratic MLP: homogeneous of degree 3):
    # loss push = -<theta, grad L> = (2 * 3 / (N p)) sum (y - f) . f  over training inputs,
    # weight-decay pull = wd * ||theta||^2
    Xtr, ytr = X[r["train_idx"]], y[r["train_idx"]]
    ftr = m(Xtr)
    Ytr = torch.nn.functional.one_hot(ytr, P).float()
    push = 6.0 / (Xtr.shape[0] * P) * float(((Ytr - ftr) * ftr).sum())
    pull = r["config"]["wd"] * float(w["W1"].pow(2).sum() + w["W2"].pow(2).sum())
    _, f90, f99 = freq_usage(w["W1"].numpy()[:, :P])
    h = m.pre(Xte)
    h_top = h @ (U[:, :K] @ U[:, :K].T)
    h_rest = h - h_top
    return dict(d99=d_at(ks, accs, orig, 0.99), er=effective_rank(w["W1"]),
                W1=float(w["W1"].norm()), W2=float(w["W2"].norm()), s1=float(S[0]),
                top=float(S[:K].pow(2).sum().sqrt()), rest=float(S[K:].pow(2).sum().sqrt()),
                accK=100 * float(accs[K - 1]), test=100 * float(orig),
                push=push, pull=pull, f90=f90, f99=f99,
                fc=float(fc.median()), margin=float(margin.median()),
                cross=float(m.readout(2 * h_top * h_rest).norm(dim=1).median()),
                square=float(m.readout(h_rest * h_rest).norm(dim=1).median()))


def section_stage4_widening(R, root, main, out):
    R.h("E3. Why the sufficient subspace widens in Stage 4 (weight growth x quadratic activation)")
    if main[0]["config"]["act"] != "quadratic":
        R.p("  skipped: the decomposition assumes the quadratic activation")
        return
    steps = [s for s in (20000, 30000, 50000) if s in main[0]["ckpts"]] + [main[0]["final_step"]]
    if steps[0] != 20000:
        R.p("  skipped: no checkpoint at step 20,000")
        return
    width = main[0]["ckpts"][20000]["W1"].shape[0]
    K = int(np.median(clean([d99_at(r, 20000) for r in main])))
    R.p(f"  K = {K} (median d_0.99 at step 20,000, where the sufficient subspace is narrowest)")
    R.p(f"  'size' = sqrt(sum of sigma_i^2) over the top K or the remaining {width - K} directions of W1;")
    R.p(f"  output change = median over test inputs of the L2 norm of the change in the {P} outputs")
    R.p("  caused by removing the directions beyond the top K, split into the two terms of")
    R.p("  (h_top + h_rest)^2 - h_top^2 = 2 h_top h_rest + h_rest^2")
    main_stats = {s: [widening_stats(r, s, K) for r in main] for s in steps}
    rows = [("d99", "d_0.99", 1), ("er", "ER(W1)", 1), ("W1", "||W1||_F", 0), ("W2", "||W2||_F", 0),
            ("s1", "sigma_1(W1)", 1), ("top", f"size of top {K}", 0), ("rest", f"size beyond top {K}", 0),
            ("accK", f"test acc, top-{K} truncation %", 1), ("fc", "median correct-class output", 3),
            ("margin", "median margin", 3), ("cross", "output change: 2 h_top h_rest", 3),
            ("square", "output change: h_rest^2", 3)]

    def ms(xs, nd):
        m_, s_, _ = mstd(xs)
        return f"{m_:.{nd}f} ± {s_:.{nd}f}"

    R.p(f"\n  [main runs, n={len(main)}]")
    R.p(f"  {'':<34}" + "".join(f"{s:>18}" for s in steps))
    for key, lab, nd in rows:
        R.p(f"  {lab:<34}" + "".join(f"{ms([d[key] for d in main_stats[s]], nd):>18}" for s in steps))
    sci = lambda key: "".join(f"{np.mean([d[key] for d in main_stats[s]]):>18.2e}" for s in steps)
    R.p(f"  {'loss push on ||theta||^2 (GD terms)':<34}" + sci("push"))
    R.p(f"  {'weight-decay pull wd*||theta||^2':<34}" + sci("pull"))
    R.p("  (gradient-flow terms; under plain gradient descent the norm grows only if push > pull,")
    R.p("   so growth with push << pull comes from AdamW's normalised updates)")

    R.p("\n  [weight decay: step 20,000 -> final checkpoint]")
    R.p(f"  {'weight decay':<14} {'n':>2} {'d_0.99':>14} {'ER(W1)':>16} {'||W1||_F':>16} "
        f"{'sigma_1':>14} {f'size beyond {K}':>16} {'final test %':>13} {'#freq 90/99% (end)':>20}")
    base = [r for r in main if r["seed"] < 5]
    wd_stats = {}
    for name, e in [("0", "wd_0"), ("1e-3", "wd_0.001"), ("1e-2 (main)", None),
                    ("5e-2", "wd_0.05"), ("1e-1", "wd_0.1")]:
        rs = [r for r in (base if e is None else load_exp(root, e)) if 20000 in r["ckpts"]]
        if not rs:
            continue
        a = [widening_stats(r, 20000, K) for r in rs]
        b = [widening_stats(r, r["final_step"], K) for r in rs]
        wd_stats[name] = dict(start=a, end=b)
        arrow = lambda key, nd: (f"{mstd([d[key] for d in a])[0]:.{nd}f} -> "
                                 f"{mstd([d[key] for d in b])[0]:.{nd}f}")
        R.p(f"  {name:<14} {len(rs):>2} {arrow('d99', 1):>14} {arrow('er', 1):>16} {arrow('W1', 0):>16} "
            f"{arrow('s1', 0):>14} {arrow('rest', 0):>16} {mstd([d['test'] for d in b])[0]:>13.1f} "
            f"{mstd([d['f90'] for d in b])[0]:>9.1f} / {mstd([d['f99'] for d in b])[0]:<8.1f}")
    R.data["stage4_widening"] = dict(K=K, main={str(s): v for s, v in main_stats.items()}, wd=wd_stats)

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    for key, lab in (("top", f"top {K} directions"), ("rest", f"remaining {width - K} directions")):
        axs[0].errorbar(steps, [mstd([d[key] for d in main_stats[s]])[0] for s in steps],
                        yerr=[mstd([d[key] for d in main_stats[s]])[1] for s in steps],
                        marker="o", capsize=3, label=lab)
    axs[0].set_xscale("log")
    axs[0].set_xlabel("step")
    axs[0].set_ylabel("size, sqrt(sum sigma_i^2)")
    axs[0].legend(fontsize=8)
    names = list(wd_stats)
    x = np.arange(len(names))
    for key, lab, mk in (("start", "step 20,000", "o"), ("end", "final", "s")):
        axs[1].errorbar(x, [mstd([d["d99"] for d in wd_stats[n][key]])[0] for n in names],
                        yerr=[mstd([d["d99"] for d in wd_stats[n][key]])[1] for n in names],
                        marker=mk, capsize=3, ls="none", label=lab)
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(names)
    axs[1].set_xlabel("weight decay")
    axs[1].set_ylabel("d_0.99 of W1")
    axs[1].legend(fontsize=8)
    savefig(out, "fig_stage4_widening.png")


# ================================================================ F. stability
def section_stability(R, runs, dims):
    R.h("F. Temporal stability of the sufficient subspace (App. C)")
    d99 = [d for d in dims[0.99] if d is not None]
    if not d99:
        R.p("  no d_0.99 available -- skipped")
        return
    k = int(round(np.median(d99)))
    res = {}
    for r in runs:
        fin = r["ckpts"][r["final_step"]]["W1"].numpy()
        Uf, _, Vhf = np.linalg.svd(fin, full_matrices=False)
        for s, w in sorted(r["ckpts"].items()):
            if s == r["final_step"]:
                continue
            U, _, Vh = np.linalg.svd(w["W1"].numpy(), full_matrices=False)
            res.setdefault(s, {"left": [], "right": []})
            res[s]["left"].append(subspace_cos(U[:, :k], Uf[:, :k]))
            res[s]["right"].append(subspace_cos(Vh[:k].T, Vhf[:k].T))
    bl, br = random_baseline(k, fin.shape[0]), random_baseline(k, fin.shape[1])
    tg = clean([grokking_onset(r["hist"]["steps"], r["hist"]["test_acc"]) for r in runs])
    R.p(f"  k = {k} (median d_0.99);  mean t_g = {tg.mean() if len(tg) else float('nan'):.0f}")
    R.p(f"  {'step':>7}  {'hidden-space subspace (C)':>28}  {'input-space subspace':>24}")
    for s in sorted(res):
        R.p(f"  {s:>7}  {fmt(res[s]['left'], 3):>28}  {fmt(res[s]['right'], 3):>24}")
    R.p(f"  random baseline     {bl[0]:.3f} ± {bl[1]:.3f} (k in R^{fin.shape[0]})"
        f"   {br[0]:.3f} ± {br[1]:.3f} (k in R^{fin.shape[1]})")
    R.data["stability"] = {str(s): v for s, v in res.items()}


# ================================================================ G. overlap across seeds
def weight_basis(W1, k=60):
    Wc = W1 - W1.mean(0)
    return np.linalg.svd(Wc, full_matrices=False)[2][:k].T          # (2p, k)


def freq_power(W1, p=P):
    pw = (np.abs(np.fft.fft(W1[:, :p], axis=1)) ** 2).mean(0)
    return pw


def freq_usage(M, p=P):
    """How many frequencies a weight matrix uses. M has one row per unit and p
    columns (one operand / token axis). Returns (share of non-DC power in the
    top 5 frequencies, #frequencies for 90% of it, #frequencies for 99%),
    with f and p-f counted as one frequency. Same function in
    analyze_transformer.py."""
    pw = (np.abs(np.fft.fft(M, axis=1)) ** 2).mean(0)
    fs = np.arange(1, (p - 1) // 2 + 1)
    pr = np.sort(pw[fs] + pw[p - fs])[::-1]
    c = np.cumsum(pr) / pr.sum()
    return float(c[4]), int(np.argmax(c >= 0.90) + 1), int(np.argmax(c >= 0.99) + 1)


def key_freqs(W1, p=P, K=5):
    pw = freq_power(W1, p)
    fs = np.arange(1, (p - 1) // 2 + 1)
    pair = pw[fs] + pw[p - fs]                                      # f and p-f are one frequency
    return sorted(int(f) for f in fs[np.argsort(pair)[::-1][:K]])


def fourier_basis(freqs, p=P):
    n = np.arange(p)
    M = np.stack([v for f in freqs for v in (np.cos(2 * np.pi * f * n / p),
                                              np.sin(2 * np.pi * f * n / p))], axis=1)
    return orth(M)                                                  # (p, 2K)


def fourier_basis_original(W1, p=P, k=10):
    """The paper's original construction (kept to show its effect): the top-k
    frequency *indices* include both f and p-f, so after QR several of the
    k kept columns are numerically arbitrary."""
    mag = np.abs(np.fft.fft(W1[:, :p], axis=1)).mean(0)
    top = np.argsort(mag)[::-1][:k]
    n = np.arange(p)
    B = [v / (np.linalg.norm(v) + 1e-12) for f in top
         for v in (np.cos(2 * np.pi * f * n / p), np.sin(2 * np.pi * f * n / p))]
    return np.linalg.qr(np.stack(B[:2 * k], 1))[0][:, :k]


def section_overlap(R, runs, out, dims=None):
    R.h("G. Cross-seed subspace overlap (Sec. 5)")
    W = {r["seed"]: r["ckpts"][r["final_step"]]["W1"].numpy() for r in runs}
    seeds = sorted(W)

    # how many frequencies does each seed use? (2 per frequency: cos and sin)
    use = {s: freq_usage(W[s][:, :P]) for s in seeds}
    d99 = dict(zip([r["seed"] for r in runs], dims[0.99])) if dims else {}
    R.p("  frequencies used by W1 (operand a), f and p-f merged; compare 2 x #freq with d_0.99")
    for s in seeds:
        t5, n90, n99 = use[s]
        R.p(f"    seed {s:>2}: top-5 share {t5:.2f}   #freq for 90% of power {n90:>2}   "
            f"for 99% {n99:>2}   d_0.99 = {d99.get(s)}")
    R.p(f"  top-5 share {fmt([use[s][0] for s in seeds], 2)};  #freq (90%) {fmt([use[s][1] for s in seeds], 1)};"
        f"  #freq (99%) {fmt([use[s][2] for s in seeds], 1)}")
    for i, lab in ((1, "90%"), (2, "99%")):
        pairs_d = [(2 * use[s][i], d99[s]) for s in seeds if d99.get(s) is not None]
        if len(pairs_d) > 2:
            a = np.array(pairs_d, dtype=float)
            ok = a[:, 0].std() > 0 and a[:, 1].std() > 0
            rho = stats.pearsonr(a[:, 0], a[:, 1])[0] if ok else float("nan")
            R.p(f"  2 x #freq ({lab}) = {a[:, 0].mean():.1f}  vs  d_0.99 = {a[:, 1].mean():.1f};  "
                f"Pearson r across seeds = {rho:.2f}")
    R.data["freq_usage"] = {str(s): use[s] for s in seeds}

    if len(seeds) < 2:
        R.p("  need >= 2 seeds -- overlap skipped")
        return
    wb = {s: weight_basis(W[s]) for s in seeds}
    kf = {s: key_freqs(W[s]) for s in seeds}
    fb = {s: fourier_basis(kf[s]) for s in seeds}
    fo = {s: fourier_basis_original(W[s]) for s in seeds}
    pairs = list(combinations(seeds, 2))
    wov = [subspace_cos(wb[a], wb[b]) for a, b in pairs]
    fov = [subspace_cos(fb[a], fb[b]) for a, b in pairs]
    foo = [subspace_cos(fo[a], fo[b]) for a, b in pairs]
    jac = [len(set(kf[a]) & set(kf[b])) / len(set(kf[a]) | set(kf[b])) for a, b in pairs]
    shared = [int((np.linalg.svd(wb[a].T @ wb[b], compute_uv=False) > 0.9).sum()) for a, b in pairs]
    rb_w, rb_f = random_baseline(60, 2 * P), random_baseline(10, P)
    rng = np.random.default_rng(1)
    rand_shared = np.mean([(np.linalg.svd(orth(rng.standard_normal((2 * P, 60))).T
                                          @ orth(rng.standard_normal((2 * P, 60))),
                                          compute_uv=False) > 0.9).sum() for _ in range(300)])
    R.p(f"  {len(seeds)} seeds, {len(pairs)} pairs")
    R.p(f"  key frequencies per seed (top 5, f and p-f merged):")
    for s in seeds:
        R.p(f"    seed {s:>2}: {kf[s]}   (DC share of power: {freq_power(W[s])[0] / freq_power(W[s]).sum():.3f})")
    R.p(f"  weight-subspace overlap (top-60, input space): {fmt(wov, 3)}   "
        f"random: {rb_w[0]:.3f} ± {rb_w[1]:.3f}")
    R.p(f"  Fourier-subspace overlap (5 key freqs, 10 dims): {fmt(fov, 3)}   "
        f"random: {rb_f[0]:.3f} ± {rb_f[1]:.3f}")
    R.p(f"  [original construction, for comparison]: {fmt(foo, 3)}")
    R.p(f"  Jaccard overlap of key-frequency sets: {fmt(jac, 3)}")
    R.p(f"  shared directions (cos > 0.9 in top-60), per pair: {fmt(shared, 2)}   "
        f"random expectation: {rand_shared:.2f}")
    # what are the shared directions? (first pair)
    a, b = pairs[0]
    Ua, s_, _ = np.linalg.svd(wb[a].T @ wb[b])
    vecs = wb[a] @ Ua[:, s_ > 0.9]
    for i in range(vecs.shape[1]):
        v = vecs[:, i]
        desc = []
        for blk, seg in (("a", v[:P]), ("b", v[P:])):
            pw = np.abs(np.fft.fft(seg)) ** 2
            fstar = int(np.argmax(pw[1:(P - 1) // 2 + 1]) + 1)
            desc.append(f"{blk}: DC {pw[0] / pw.sum():.2f}, f={fstar} {(pw[fstar] + pw[P - fstar]) / pw.sum():.2f}")
        R.p(f"    shared direction {i} (seeds {a},{b}, cos={s_[i]:.3f}): " + "; ".join(desc))
    R.data["overlap"] = dict(weight=wov, fourier=fov, fourier_original=foo, jaccard=jac,
                             shared=shared, rand_weight=rb_w, rand_fourier=rb_f,
                             rand_shared=rand_shared, key_freqs=kf)
    # Fig. 5: Fourier power spectrum per seed
    fs = np.arange(1, (P - 1) // 2 + 1)
    show = seeds[:6]
    fig, axs = plt.subplots(1, len(show), figsize=(3 * len(show), 2.6), sharey=True)
    axs = np.atleast_1d(axs)
    for ax, s in zip(axs, show):
        pw = freq_power(W[s])
        ax.bar(fs, (pw[fs] + pw[P - fs]) / pw.sum(), width=0.9)
        ax.set_title(f"seed {s}")
        ax.set_xlabel("frequency")
    axs[0].set_ylabel("share of power")
    savefig(out, "fig_fourier_spectra_seeds.png")


# ================================================================ H. patching sanity
def section_patching(R, run):
    R.h("H. Activation patching sanity check (MLP, one seed)")
    X, y, _, te = run_data(run)
    model = final_model(run)
    Xte, yte = X[te], y[te]
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():
        h_c = model.pre(Xte)
        z_c = model.act(h_c)
        R.p(f"  clean accuracy {100 * accuracy(model(Xte), yte):.2f}%")
        for s in (0.1, 0.3, 0.5):
            Xn = Xte + s * torch.randn(Xte.shape, generator=g)
            corrupt = accuracy(model(Xn), yte)
            pre = accuracy(model.readout(model.act(h_c)), yte)
            post = accuracy(model.readout(z_c), yte)
            R.p(f"  sigma={s}: corrupted {100 * corrupt:.2f}%  pre-patched {100 * pre:.2f}%  "
                f"post-patched {100 * post:.2f}%")
    R.p("  (with one hidden layer, patching the whole pre- or post-activation restores the "
        "clean output by construction)")


# ================================================================ I. transplant
def post_act(model, X):
    with torch.no_grad():
        return model.act(model.pre(X))


def transplant(A, B, X_fit, X_ev, y_ev):
    ZA_fit, ZB_fit, ZA_ev = post_act(A, X_fit), post_act(B, X_fit), post_act(A, X_ev)
    with torch.no_grad():
        un = accuracy(B.readout(ZA_ev), y_ev)
        U, _, Vt = torch.linalg.svd(ZA_fit.T @ ZB_fit)
        al = accuracy(B.readout(ZA_ev @ (U @ Vt)), y_ev)
    return un, al


def section_transplant(R, runs, mem_runs):
    R.h("I. Transplant patching with and without Procrustes alignment (App. D)")
    R.p("  fit the rotation on model A's training inputs, evaluate on A's test inputs,")
    R.p("  against the true (a+b) mod p labels.")
    by_seed = {r["seed"]: r for r in runs}
    true_y = run_data(runs[0])[1]

    def eval_pair(rA, mA, mB, label, store):
        X, _, tr, te = run_data(rA)
        un, al = transplant(mA, mB, X[tr], X[te], true_y[te])
        store.append((un, al))

    groups = {}
    seeds = sorted(by_seed)
    g = groups.setdefault("grokked -> grokked (other seed)", [])
    for a, b in zip(seeds[0::2], seeds[1::2]):
        mA, mB = final_model(by_seed[a]), final_model(by_seed[b])
        eval_pair(by_seed[a], mA, mB, "", g)
        eval_pair(by_seed[b], mB, mA, "", g)
    for mr in mem_runs:
        s = mr["seed"]
        if s not in by_seed:
            continue
        mG, mM = final_model(by_seed[s]), final_model(mr)
        eval_pair(by_seed[s], mG, mM, "", groups.setdefault("grokked -> memorizing (random labels)", []))
        eval_pair(by_seed[s], mM, mG, "", groups.setdefault("memorizing (random labels) -> grokked", []))
    for s, r in by_seed.items():
        h = r["hist"]
        cand = [st for st in r["ckpts"] if st != r["final_step"] and st in set(h["steps"].tolist())
                and h["train_acc"][list(h["steps"]).index(st)] >= 0.99
                and h["test_acc"][list(h["steps"]).index(st)] <= 0.5]
        if not cand:
            continue
        w = r["ckpts"][max(cand)]
        groups.setdefault("_memorization checkpoint (step, test acc)", []).append(
            (max(cand), float(h["test_acc"][list(h["steps"]).index(max(cand))])))
        mM = model_from_weights(w["W1"], w["W2"], r["config"]["act"])
        mG = final_model(r)
        eval_pair(r, mG, mM, "", groups.setdefault("grokked -> same seed, memorization-phase checkpoint", []))
        eval_pair(r, mM, mG, "", groups.setdefault("memorization-phase checkpoint -> grokked", []))
    for name, vals in groups.items():
        if vals and not name.startswith("_"):
            R.p(f"  {name:<56} unaligned {fmt([100 * v[0] for v in vals], 2)} %   "
                f"aligned {fmt([100 * v[1] for v in vals], 2)} %")
    R.p(f"  chance = {100 / P:.2f} %")
    mem = groups.get("_memorization checkpoint (step, test acc)", [])
    if mem:
        R.p(f"  memorization-phase checkpoint = last checkpoint with train acc >= 99% and test acc <= 50%:")
        R.p(f"    steps {sorted(set(v[0] for v in mem))}, test accuracy there {fmt([100 * v[1] for v in mem], 1)} %")
    R.data["transplant"] = groups


# ================================================================ J. variants
def stage4_drop(h):
    er = np.asarray(h["W1_erank"])
    tg = grokking_onset(h["steps"], h["test_acc"])
    if tg is None:
        return None
    i = list(h["steps"]).index(tg)
    peak = er[i:].max()
    return float((peak - er[-1]) / peak)


def describe(runs):
    tg = [grokking_onset(r["hist"]["steps"], r["hist"]["test_acc"]) for r in runs]
    return dict(n=len(runs), grokked=len(clean(tg)), tg=tg,
                final_test=[float(r["hist"]["test_acc"][-1]) for r in runs],
                final_train=[float(r["hist"]["train_acc"][-1]) for r in runs],
                final_er=[float(r["hist"]["W1_erank"][-1]) for r in runs],
                final_ipr=[float(r["hist"]["fourier_ipr"][-1]) for r in runs],
                stage4=[stage4_drop(r["hist"]) for r in runs])


def line(name, d):
    return (f"  {name:<14} grokked {d['grokked']}/{d['n']}  t_g {fmt(d['tg'], 0)}  "
            f"final test {fmt([100 * v for v in d['final_test']], 1)}%  "
            f"final ER {fmt(d['final_er'], 1)}  Stage-4 ER drop {fmt([100 * v if v is not None else None for v in d['stage4']], 1)}%  "
            f"final Fourier IPR {fmt(d['final_ipr'], 3)}")


def mean_curve(runs, key):
    n = min(len(r["hist"][key]) for r in runs)
    return runs[0]["hist"]["steps"][:n], np.mean([r["hist"][key][:n] for r in runs], 0)


def section_variants(R, root, main, out):
    R.h("J. Variants: loss, weight decay, activation, rank interventions")
    base = [r for r in main if r["seed"] < 5]
    data = {}

    def show(title, groups, fig=None, keys=("test_acc", "W1_erank")):
        groups = [(n, rs) for n, rs in groups if rs]
        if len(groups) < 2:
            return
        R.p(f"\n  [{title}]")
        for n, rs in groups:
            d = describe(rs)
            data[f"{title}|{n}"] = d
            R.p(line(n, d))
        if fig:
            fig_, axs = plt.subplots(1, len(keys), figsize=(6 * len(keys), 3.6))
            for ax, key in zip(np.atleast_1d(axs), keys):
                for n, rs in groups:
                    st, m = mean_curve(rs, key)
                    ax.plot(st, m, label=n)
                ax.set_xlabel("step")
                ax.set_ylabel(key)
                ax.legend(fontsize=7)
            savefig(out, fig)

    R.p("  (final Fourier IPR: 0.5 = every unit uses one frequency; ~0.03 = no periodic structure)")
    show("loss", [("MSE (main)", base), ("CE lr=1e-3", load_exp(root, "loss_ce")),
                  ("CE lr=3e-3", load_exp(root, "loss_ce_lr0.003")),
                  ("CE lr=1e-2", load_exp(root, "loss_ce_lr0.01"))],
         "fig_loss_comparison.png", keys=("test_acc", "W1_erank", "fourier_ipr"))
    wds = [("wd=0", "wd_0"), ("wd=1e-3", "wd_0.001"), ("wd=1e-2 (main)", None),
           ("wd=5e-2", "wd_0.05"), ("wd=1e-1", "wd_0.1")]
    show("weight decay", [(n, base if e is None else load_exp(root, e)) for n, e in wds],
         "fig_wd_sweep.png")
    show("activation", [("quadratic (main)", base), ("ReLU", load_exp(root, "act_relu")),
                        ("GELU", load_exp(root, "act_gelu"))], "fig_activation_comparison.png",
         keys=("test_acc", "W1_erank", "fourier_ipr"))
    nucs = [("lambda=0 (main)", base)] + [(f"lambda={l:.0e}", load_exp(root, f"nuc_{l:.0e}"))
                                         for l in (1e-8, 3e-8, 1e-7, 3e-7, 1e-6)]
    show("nuclear-norm penalty on W1 from step 0", nucs, "fig_rank_penalty.png")
    # did the penalty compress W1 *before* grokking? compare ER at the baseline's t_g
    tg0 = clean([grokking_onset(r["hist"]["steps"], r["hist"]["test_acc"]) for r in base])
    if len(tg0) and any(rs for _, rs in nucs[1:]):
        ref = int(np.round(tg0.mean() / base[0]["config"]["log_every"]) * base[0]["config"]["log_every"]) - \
            2 * base[0]["config"]["log_every"]
        R.p(f"    ER(W1) shortly before the baseline's mean t_g (step {ref}):")
        for n, rs in nucs:
            vals = [r["hist"]["W1_erank"][list(r["hist"]["steps"]).index(ref)]
                    for r in rs if ref in set(r["hist"]["steps"].tolist())]
            if vals:
                R.p(f"      {n:<16} {fmt(vals, 1)}")
    ranks = [(f"rank {r}", load_exp(root, f"rank_{r}")) for r in (8, 16, 32, 48, 64, 80, 96)]
    ranks.append(("AB, rank 128", load_exp(root, "rank_128")))      # factorised but full rank
    show("hard bottleneck W1 = AB", ranks + [("W1, full (main)", [r for r in main if r["seed"] < 3])],
         "fig_rank_bottleneck.png")
    R.p("    'AB, rank 128' is factorised like the bottlenecks but has full rank: if it groks as early")
    R.p("    as rank 32-96, the speed-up comes from the factorisation, not from the rank limit.")
    fac, unf = load_exp(root, "rank_128"), [r for r in main if r["seed"] < 3]
    if fac and unf:
        R.p("    effective rank of W1 before and after grokking: factorised r=128 vs unfactorised (seeds 0-2)")
        R.p(f"    {'step':>7} {'AB, rank 128':>22} {'W1, full (main)':>22}")
        er_at = lambda r, s0: float(np.asarray(r["hist"]["W1_erank"])[np.argmin(np.abs(np.asarray(r["hist"]["steps"]) - s0))])
        for s0 in (0, 500, 1000, 1500, 2000, 3000, 5000, 10000, 20000):
            R.p(f"    {s0:>7} {fmt([er_at(r, s0) for r in fac], 1):>22} {fmt([er_at(r, s0) for r in unf], 1):>22}")
    R.data["variants"] = data


# ================================================================ K. main-run figures
def smooth_rate(steps, v, k=5):
    """d/dt of the k-point moving average, as in steepest_drop (Table 3). The series is padded
    with its edge values instead of zeros, so the first and last points are not distorted;
    every interior point equals the Table 3 measure."""
    v = np.asarray(v, dtype=float)
    sm = np.convolve(np.pad(v, k // 2, mode="edge"), np.ones(k) / k, mode="valid")
    return np.gradient(sm, np.asarray(steps, dtype=float))


def place_stage_labels(fig, ax, bounds, labels, pts, bold=None, max_cover=2):
    """One single-line boxed label with an arrow per band. Inside the panel: Stage 1 top left,
    Stage 3 at mid-height in its band, Stage 4 at the top of its band. A label that would cover
    more than `max_cover` data points or block Stage 2's arrow, and always Stage 2 (its band is
    narrow), goes instead in a row just above the panel, left to right without overlaps, with its arrow down into its band."""
    xc = [(bounds[i] + bounds[i + 1]) / 2 for i in range(4)]
    box = dict(boxstyle="round,pad=0.3", fc="white", ec="#9cc3ec", lw=1.0)
    arrow = dict(arrowstyle="-|>", color="0.3", lw=0.8, shrinkA=1, shrinkB=1)
    common = dict(xycoords=("data", "axes fraction"), textcoords=("data", "axes fraction"),
                  fontsize=7, bbox=box, arrowprops=arrow, zorder=5, annotation_clip=False)
    r = fig.canvas.get_renderer()
    fig.canvas.draw()
    axb = ax.get_window_extent(r)
    to_disp, to_data = ax.transData.transform, ax.transData.inverted().transform
    pts_d = to_disp(pts)
    inside = {0: (0.95, 0.70), 2: (0.58, 0.32), 3: (0.95, 0.72)}     # stage: (text top, arrow tip)
    outside = [1]
    for i, (top, tip) in inside.items():
        ann = ax.annotate(labels[i], xy=(xc[i], tip), xytext=(xc[i], top), ha="center", va="top",
                          fontweight="bold" if i == bold else "normal", **common)
        fig.canvas.draw()
        bb = ann.get_bbox_patch().get_window_extent(r)
        dx = max(0.0, axb.x0 + 4 - bb.x0) if i == 0 else 0.0           # Stage 1 flush left
        dx -= max(0.0, bb.x1 + dx - (axb.x1 - 4))
        if dx:
            ann.xyann = (to_data((to_disp((xc[i], 0))[0] + dx, 0))[0], top)
            bb = bb.translated(dx, 0)
        covered = int(((pts_d[:, 0] > bb.x0) & (pts_d[:, 0] < bb.x1) &
                       (pts_d[:, 1] > bb.y0) & (pts_d[:, 1] < bb.y1)).sum())
        x2 = to_disp((xc[1], 0))[0]                     # Stage 2's arrow comes down at x2,
        y2 = axb.y0 + 0.86 * axb.height                 # from the top edge to y2
        blocks = bb.x0 - 6 < x2 < bb.x1 + 6 and bb.y1 > y2 - 6
        if covered > max_cover or blocks:
            ann.remove()
            outside.append(i)
    y_row = 1 + 3 / axb.height                                         # 3 px above the panel
    anns = {i: ax.annotate(labels[i], xy=(xc[i], 0.86), xytext=(xc[i], y_row), ha="center",
                           va="bottom", fontweight="bold" if i == bold else "normal", **common)
            for i in sorted(outside)}
    fig.canvas.draw()
    w = {i: a.get_bbox_patch().get_window_extent(r).width for i, a in anns.items()}
    x_b2 = to_disp((bounds[1], 0))[0]
    pos = {1: x_b2 + w[1] / 2}                  # Stage 2: left edge above the start of its band
    if 0 in anns:                               # Stage 1: just left of Stage 2 (may extend into
        pos[0] = x_b2 - 6 - w[0] / 2            # the margin left of the panel)
        shift = max(0.0, fig.bbox.x0 + 2 - (pos[0] - w[0] / 2))
        pos[0] += shift
        pos[1] += shift
    right = pos[1] + w[1] / 2
    for i in sorted(anns):                      # any other label: to the right, without overlaps
        if i not in pos:
            pos[i] = min(max(to_disp((xc[i], 0))[0], right + 6 + w[i] / 2), axb.x1 - w[i] / 2)
            right = pos[i] + w[i] / 2
    for i, cx in pos.items():
        anns[i].xyann = (to_data((cx, 0))[0], y_row)
    fig.canvas.draw()


def section_figures(R, run, out):
    R.h("K. Figures from one main run")
    h, st = run["hist"], run["hist"]["steps"]
    panels = [("train_loss", "test_loss", "loss"), ("W1_norm", "W2_norm", "weight norm"),
              ("W1_erank", "W2_erank", "effective rank"), ("train_acc", "test_acc", "accuracy"),
              ("grad_norm", None, "gradient norm"), ("W1_srank", "W2_srank", "stable rank"),
              ("W1_l0", None, "L0 sparsity (W1)"), ("W1_hoyer", None, "Hoyer sparsity (W1)"),
              ("fourier_ipr", None, "Fourier IPR (W1)")]
    fig, axs = plt.subplots(3, 3, figsize=(13, 10))
    for ax, (a, b, t) in zip(axs.flat, panels):
        ax.plot(st, h[a], label=a)
        if b:
            ax.plot(st, h[b], label=b)
        ax.set_title(t)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    savefig(out, "fig_training_dynamics.png")

    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(st, h["fourier_ipr"], color="tab:blue", label="Fourier IPR")
    ax2 = ax.twinx()
    ax2.plot(st, h["test_acc"], color="tab:red", label="test accuracy")
    ax.set_xlabel("step")
    ax.set_ylabel("IPR")
    ax2.set_ylabel("test accuracy")
    savefig(out, "fig_ipr_vs_accuracy.png")

    plt.figure(figsize=(6, 3.6))
    plt.plot(st, h["fourier_erank"], label="Fourier-space ER (W1)")
    plt.plot(st, h["W1_erank"], label="weight-space ER (W1)")
    plt.xlabel("step")
    plt.legend()
    savefig(out, "fig_fourier_vs_weight_rank.png")

    # Rate of change of ER(W1) and test accuracy, shaded by the four stages of Section 4.
    # Boundaries from the data: Stage 2 starts at the first logged step with test accuracy
    # above 5% (chance ~1%), Stage 3 at t_g, Stage 4 at t_c.
    st_a = np.asarray(st, dtype=float)
    acc = np.asarray(h["test_acc"], dtype=float)
    der = smooth_rate(st_a, h["W1_erank"])            # 5-point moving average, as in Table 3
    dacc = smooth_rate(st_a, acc)
    tg = grokking_onset(st, h["test_acc"])
    tc = compression_onset(st, h["W1_erank"])
    above = np.where(acc > 0.05)[0]
    t2 = int(st_a[above[0]]) if len(above) else None
    if None in (t2, tg, tc):
        R.p("  stage boundaries undefined for this run; fig_rank_derivatives.png not drawn")
    else:
        bounds = [0, t2, tg, tc, int(st_a[-1])]
        names = ["Stage 1: Memorization", "Stage 2: Grokking transition",
                 "Stage 3: Consolidation", "Stage 4: Late compression"]
        R.p(f"  fig_rank_derivatives.png stage boundaries (seed {run['seed']}): "
            f"Stage 2 from {t2} (test acc > 5%), Stage 3 from t_g = {tg}, Stage 4 from t_c = {tc}")
        shades = ["#f7faff", "#eef4ff", "#e5eeff", "#dce8fb"]     # light, as in the original figure
        fig, axs = plt.subplots(2, 1, figsize=(7.5, 6.0))   # full text width in the main text
        for ax, y, (pos_lab, neg_lab), title, ylab, dot, legloc in (
                (axs[0], der, ("Rank Increase", "Rank Decrease"), "Rate of Rank Evolution",
                 "Rank Change Rate\n(dER/dt)", None, "lower right"),
                (axs[1], dacc, ("Improving", "Degrading"), "Rate of Accuracy Evolution",
                 "Accuracy Change Rate\n(dAcc/dt)", "tab:red", "upper right")):
            for i in range(4):
                ax.axvspan(bounds[i], bounds[i + 1], color=shades[i], zorder=0)
            for bd in bounds:
                ax.axvline(bd, color="0.45", ls="--", lw=0.8, zorder=1)
            x, y = st_a[2:-2], y[2:-2]          # first/last 2 points: moving-average edge effects
            ax.fill_between(x, y, 0, where=y >= 0, color="tab:green", alpha=0.25, label=pos_lab)
            ax.fill_between(x, y, 0, where=y < 0, color="tab:red", alpha=0.25, label=neg_lab)
            colors = dot or np.where(y >= 0, "tab:green", "#5b8ae6")
            ax.scatter(x, y, s=14, c=colors, alpha=0.75, zorder=3)
            ax.axhline(0, color="k", lw=0.6)
            ax.grid(alpha=0.3)
            ax.set_title(title, fontweight="bold", pad=22)    # room for the label row above
            ax.set_xlabel("Training Steps")
            ax.set_ylabel(ylab)
            hs, ls = ax.get_legend_handles_labels()
            order = [1, 0] if ax is axs[0] else [0, 1]      # original: "Rank Decrease" first
            ax.legend([hs[k] for k in order], [ls[k] for k in order], loc=legloc, fontsize=9)
        fig.tight_layout()
        for ax, y in ((axs[0], der), (axs[1], dacc)):
            place_stage_labels(fig, ax, bounds, names, np.column_stack([st_a[2:-2], y[2:-2]]), bold=2)
        fig.savefig(os.path.join(out, "fig_rank_derivatives.png"), dpi=200)
        plt.close(fig)

    X, y, _, te = run_data(run)
    model = final_model(run)
    with torch.no_grad():
        hpre = model.pre(X).numpy()
        z = model.act(model.pre(X)).numpy()
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, M, t in ((axs[0], hpre, "pre-activations"), (axs[1], z, "post-activations")):
        Mc = M - M.mean(0)
        U, S, _ = np.linalg.svd(Mc, full_matrices=False)
        ev = S ** 2 / (S ** 2).sum()
        ax.scatter(U[:, 0] * S[0], U[:, 1] * S[1], c=y.numpy(), s=2, cmap="hsv")
        ax.set_title(f"{t}: PC1 {100 * ev[0]:.1f}%, PC2 {100 * ev[1]:.1f}%")
    savefig(out, "fig_pca_structure.png")

    Xte, yte = X[te], y[te]
    W1, W2 = model.W1_full.detach(), model.W2.detach()
    g = torch.Generator().manual_seed(0)
    sig = np.linspace(0, 1.5, 16)
    accs = []
    for s in sig:
        a = []
        for _ in range(5):
            m2 = model_from_weights(W1 + s * W1.std() * torch.randn(W1.shape, generator=g),
                                    W2 + s * W2.std() * torch.randn(W2.shape, generator=g),
                                    run["config"]["act"])
            with torch.no_grad():
                a.append(accuracy(m2(Xte), yte))
        accs.append(np.mean(a))
    R.p("  weight perturbation (noise sd = sigma x sd of each matrix): " +
        ", ".join(f"{s:.1f}:{100 * a:.0f}%" for s, a in zip(sig[::3], accs[::3])))
    plt.figure(figsize=(5, 3.5))
    plt.plot(sig, accs, "o-")
    plt.xlabel("sigma (relative to weight sd)")
    plt.ylabel("test accuracy")
    savefig(out, "fig_noise_robustness.png")

    Wn = W1.numpy()
    Fa, Fb = np.fft.fft(Wn[:, :P], axis=1), np.fft.fft(Wn[:, P:], axis=1)
    fs = np.arange(1, (P - 1) // 2 + 1)
    fstar = fs[np.argmax(np.abs(Fa[:, fs]) ** 2 + np.abs(Fb[:, fs]) ** 2, axis=1)]
    pa = np.angle(Fa[np.arange(len(fstar)), fstar])
    pb = np.angle(Fb[np.arange(len(fstar)), fstar])
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
    axs[0].scatter(pa, pb, s=8)
    axs[0].set_xlabel("phase on operand a")
    axs[0].set_ylabel("phase on operand b")
    axs[1].hist(np.mod(pa - pb + np.pi, 2 * np.pi) - np.pi, bins=36)
    axs[1].set_xlabel("phase difference a - b")
    savefig(out, "fig_phase_alignment.png")


# ================================================================ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(args.runs, "analysis_mlp")
    os.makedirs(out, exist_ok=True)
    torch.set_grad_enabled(False)

    R = Report()
    main_runs = load_exp(args.runs, "main")
    neg = load_exp(args.runs, "random_labels")
    if not main_runs:
        raise SystemExit(f"no runs in {args.runs}/mlp/main")
    R.p(f"main runs: {len(main_runs)} seeds; config: {main_runs[0]['config']}")

    section_onsets(R, main_runs)
    section_xcorr(R, main_runs, out)
    section_granger(R, main_runs)
    section_onset_sensitivity(R, main_runs, neg)
    dims = section_causal_dims(R, main_runs, out)
    section_dc_over_training(R, main_runs, out)
    section_margin_free(R, main_runs)
    section_stage4_widening(R, args.runs, main_runs, out)
    section_stability(R, main_runs, dims)
    section_overlap(R, main_runs, out, dims)
    section_patching(R, main_runs[0])
    section_transplant(R, main_runs, neg)
    section_variants(R, args.runs, main_runs, out)
    section_figures(R, main_runs[0], out)

    # Fig. 7a: t_c vs t_g
    rows = R.data["onsets"]
    pts = [(w["tg"], w["tc"]) for w in rows if w["tg"] is not None and w["tc"] is not None]
    if pts:
        a = np.array(pts)
        plt.figure(figsize=(4.5, 4.5))
        plt.scatter(a[:, 0], a[:, 1])
        lim = [a.min() * 0.9, a.max() * 1.1]
        plt.plot(lim, lim, "k--", lw=1, label="t_c = t_g")
        plt.xlabel("t_g (steps)")
        plt.ylabel("t_c (steps)")
        plt.legend()
        savefig(out, "fig_leadlag_scatter.png")

    R.save(out)
    print(f"\nwrote {out}/summary.txt, summary.json and figures")


if __name__ == "__main__":
    main()
