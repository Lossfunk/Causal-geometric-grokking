"""
prereg_transformer_timing.py

Applies the steepest-drop measure, unchanged from analyze_transformer.py, to
transformer seeds that were not used to define it:

  python run_transformer.py --seeds 15-29 --out results_holdout
  python prereg_transformer_timing.py --runs results_holdout

Writes <runs>/prereg_H1/summary.txt, summary.json and fig_er_aligned.png.
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from analyze_transformer import clean, fmt, grokked, load_runs, steepest_drop, t_grok

START = 4000         
USED_TO_DEFINE = set(range(15))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results_holdout")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or os.path.join(args.runs, "prereg_H1")
    os.makedirs(out, exist_ok=True)

    runs = load_runs(args.runs, 1.0)
    if not runs:
        raise SystemExit(f"no runs in {args.runs}/transformer/wd1")
    reused = sorted(USED_TO_DEFINE & {r["seed"] for r in runs})
    if reused:
        raise SystemExit(f"seeds {reused} were used to define the measure; keep only held-out seeds in {args.runs}")

    lines = [f"Steepest ER(W_in) decrease after step {START:,} minus t_g > 0",
             f"held-out seeds found: {sorted(r['seed'] for r in runs)}", ""]
    rows, excluded = [], []
    for r in runs:
        h = r["hist"]
        if not grokked(r):
            excluded.append((r["seed"], float(h["test_acc"][-1])))
            continue
        tg = t_grok(h)
        ts = steepest_drop(h["steps"], h["W_in_erank"], START)
        rows.append(dict(seed=r["seed"], tg=tg, ts=ts, lag=ts - tg))
        lines.append(f"  seed {r['seed']:>2}: t_g={tg:>6}  t_s={ts:>6}  t_s - t_g={ts - tg:+6d}")
    for s, a in excluded:
        lines.append(f"  seed {s:>2}: excluded, did not grok (final test accuracy {100 * a:.1f}%)")

    lags = clean([w["lag"] for w in rows])
    res = {"rows": rows, "excluded": excluded}
    if len(lags) > 1:
        t, p = stats.ttest_1samp(lags, 0, alternative="greater")
        k = int((lags > 0).sum())
        sign = stats.binomtest(k, len(lags), 0.5, alternative="greater").pvalue
        ci = stats.t.interval(0.95, len(lags) - 1, loc=lags.mean(), scale=stats.sem(lags))
        lines += ["",
                  f"t_s - t_g: {fmt(lags, 0, med=True)};  95% CI [{ci[0]:.0f}, {ci[1]:.0f}]",
                  f"one-sided one-sample t-test: t={t:.2f}, p={p:.2e}",
                  f"t_s > t_g in {k}/{len(lags)} seeds; exact sign test p={sign:.2e}",
                  f"Hypothesis {'SUPPORTED' if p < 0.05 else 'NOT SUPPORTED'} at alpha = 0.05",
                  "(seeds 0-14, which defined the measure: +964 +- 499 steps, 13/14 after t_g)"]
        res.update(t=float(t), p=float(p), k=k, n=len(lags), sign_p=float(sign), ci=[float(c) for c in ci],
                   mean=float(lags.mean()), sd=float(lags.std(ddof=1)))
    else:
        lines.append("fewer than two grokked seeds: test not run")

    with open(os.path.join(out, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), sharex=True)
    for r in runs:
        if not grokked(r):
            continue
        h = r["hist"]
        x = np.asarray(h["steps"]) - t_grok(h)
        axs[0].plot(x, np.asarray(h["W_in_erank"]) / h["W_in_erank"][0], lw=0.8, alpha=0.7)
        axs[1].plot(x, h["test_acc"], lw=0.8, alpha=0.7)
    for ax, lab in zip(axs, ("ER(W_in) / ER at step 0", "test accuracy")):
        ax.axvline(0, color="k", lw=0.8, ls="--")
        ax.set_xlabel("step - t_g")
        ax.set_ylabel(lab)
    plt.tight_layout()
    plt.savefig(os.path.join(out, "fig_er_aligned.png"), dpi=200)
    print("\n".join(lines))
    print(f"\nwrote {out}/summary.txt, summary.json and fig_er_aligned.png")


if __name__ == "__main__":
    main()
