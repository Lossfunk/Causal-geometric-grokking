"""
run_transformer.py -- train the one-layer transformer (paper Sec. 2.2).

Same configuration as the July notebook (grokking.ipynb, cell 4), which
follows Nanda et al. (2023): 4 heads, d_model 128, d_head 32, d_mlp 512,
ReLU, no LayerNorm, biases frozen, cross-entropy, AdamW(lr 1e-3, wd 1.0,
betas (0.9, 0.98)), 30% train split, 25k full-batch steps.

  python run_transformer.py --seeds 0-14                # T1 main runs
  python run_transformer.py --seeds 0-2 --wd 0.1        # T3 weight-decay arms (also 0.3)
  python run_transformer.py --seeds 0-29 --ckpt-every 250 --out results_dense
                                                        # same runs, a checkpoint every 250 steps
  python run_transformer.py --seeds 0-4 --factor W_in:16  # W_in = A B with rank 16 (W_E also;
                                                        # rank 128 = factorised full-rank control)

Saves <out>/transformer/wd<wd>/seed<k>.pt with a metric history logged every
250 steps (accuracy, loss, effective rank and spectral IPR of every weight
matrix, Fourier IPR of the embedding) and checkpoints at selected steps.
Requires: pip install transformer_lens
"""
import argparse
import copy
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer, HookedTransformerConfig

P = 97
EPS = 1e-10
DEFAULTS = dict(frac=0.30, steps=25_000, lr=1e-3, wd=1.0, betas=(0.9, 0.98), log_every=250)
CKPT_STEPS = [0, 2500, 5000, 7500, 9000, 10000, 11000, 12500, 15000, 20000]
MATS = ["W_E", "W_Q", "W_K", "W_V", "W_O", "W_in", "W_out", "W_U"]


def build_model(seed, device):
    cfg = HookedTransformerConfig(
        n_layers=1, d_model=128, d_head=32, n_heads=4, d_mlp=512,
        d_vocab=P + 1, n_ctx=3, act_fn="relu", normalization_type=None,
        seed=seed, device=device)
    model = HookedTransformer(cfg).to(device)
    for n, par in model.named_parameters():
        if "b_" in n:
            par.requires_grad_(False)
    return model


class LowRank(torch.nn.Module):
    """W = A B with inner dimension r, initialised so that the entries of A B have the
    variance of the original initialisation. Used through torch's parametrize, so the model
    code is unchanged and model.embed.W_E / blocks[0].mlp.W_in return A B."""

    def __init__(self, W, r, seed):
        super().__init__()
        g = torch.Generator().manual_seed(10_000 + seed)
        s = (float(W.std()) / math.sqrt(r)) ** 0.5          # Var[(AB)_ij] = r s^4 = Var[W_ij]
        self.A = torch.nn.Parameter((torch.randn(W.shape[0], r, generator=g) * s).to(W.device))
        self.B = torch.nn.Parameter((torch.randn(r, W.shape[1], generator=g) * s).to(W.device))

    def forward(self, X):
        return self.A @ self.B


FACTOR_TARGETS = {"W_E": lambda m: (m.embed, "W_E"), "W_in": lambda m: (m.blocks[0].mlp, "W_in")}


def factorize(model, name, r, seed):
    from torch.nn.utils import parametrize
    mod, attr = FACTOR_TARGETS[name](model)
    parametrize.register_parametrization(mod, attr, LowRank(getattr(mod, attr).detach(), r, seed),
                                         unsafe=True)
    mod.parametrizations[attr].original.requires_grad_(False)   # unused; A and B are trained


def plain_state(model, factor=None):
    """state_dict with a factorised matrix stored as an ordinary weight, so that build_model +
    load_state_dict (restore() in analyze_transformer.py) can read it."""
    sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if "parametrizations" not in k}
    if factor:
        mod, attr = FACTOR_TARGETS[factor[0]](model)
        key = {"W_E": "embed.W_E", "W_in": "blocks.0.mlp.W_in"}[factor[0]]
        sd[key] = getattr(mod, attr).detach().cpu().clone()
    return sd


def build_dataset(seed, frac, device):
    a = torch.arange(P).repeat_interleave(P)
    b = torch.arange(P).repeat(P)
    data = torch.stack([a, b, torch.full_like(a, P)], dim=1).to(device)
    labels = ((a + b) % P).to(device)
    torch.manual_seed(seed)
    idx = torch.randperm(P * P)
    cut = int(frac * P * P)
    return data, labels, idx[:cut].to(device), idx[cut:].to(device)


def weight_matrices(model):
    blk = model.blocks[0]
    return dict(W_E=model.embed.W_E[:-1], W_Q=blk.attn.W_Q, W_K=blk.attn.W_K,
                W_V=blk.attn.W_V, W_O=blk.attn.W_O, W_in=blk.mlp.W_in,
                W_out=blk.mlp.W_out, W_U=model.unembed.W_U)


def as2d(W):
    return W.reshape(-1, W.shape[-1]) if W.dim() > 2 else W


def _sv(W):
    return torch.linalg.svdvals(as2d(W).detach().float())


def effective_rank(W):
    s = _sv(W)
    q = s / (s.sum() + EPS)
    return float(torch.exp(-(q * torch.log(q + EPS)).sum()))


def spectral_ipr(W):
    s = _sv(W)
    q = s / (s.sum() + EPS)
    return float((q ** 4).sum())


def embed_fourier_ipr(W_E):
    """Doshi-style IPR of the DFT over tokens of each embedding dimension."""
    A = np.abs(np.fft.fft(W_E.detach().cpu().numpy(), axis=0))
    A = A / (np.linalg.norm(A, axis=0, keepdims=True) + EPS)
    return float(np.mean(np.sum(A ** 4, axis=0)))


def accuracy(logits, labels):
    return float((logits.argmax(-1) == labels).float().mean())


def train_run(seed, cfg, device, verbose=True):
    c = {**DEFAULTS, "ckpt_every": None, "factor": None, **cfg}
    model = build_model(seed, device)
    if c["factor"]:
        factorize(model, c["factor"][0], c["factor"][1], seed)
    data, labels, tr, te = build_dataset(seed, c["frac"], device)
    opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad],
                            lr=c["lr"], weight_decay=c["wd"], betas=c["betas"])
    keys = ["steps", "train_loss", "test_loss", "train_acc", "test_acc", "E_fourier_ipr"] + \
        [f"{m}_erank" for m in MATS] + [f"{m}_spectral_ipr" for m in MATS]
    hist = {k: [] for k in keys}
    ckpts = {}
    keep = {s for s in CKPT_STEPS if s < c["steps"]}
    if c["ckpt_every"]:
        keep |= set(range(0, c["steps"], c["ckpt_every"]))
    last = c["steps"] - 1
    for step in range(c["steps"]):
        model.train()
        logits = model(data[tr])[:, -1, :]
        loss = F.cross_entropy(logits, labels[tr])
        loss.backward()
        opt.step()
        opt.zero_grad()
        if step % c["log_every"] == 0 or step == last:
            model.eval()
            with torch.no_grad():
                te_logits = model(data[te])[:, -1, :]
                hist["steps"].append(step)
                hist["train_loss"].append(float(loss))
                hist["train_acc"].append(accuracy(logits, labels[tr]))
                hist["test_loss"].append(float(F.cross_entropy(te_logits, labels[te])))
                hist["test_acc"].append(accuracy(te_logits, labels[te]))
                wm = weight_matrices(model)
                for m in MATS:
                    hist[f"{m}_erank"].append(effective_rank(wm[m]))
                    hist[f"{m}_spectral_ipr"].append(spectral_ipr(wm[m]))
                hist["E_fourier_ipr"].append(embed_fourier_ipr(wm["W_E"]))
            if verbose and step % (c["log_every"] * 20) == 0:
                print(f"    step {step:>6}  test_acc={hist['test_acc'][-1]:.3f}  "
                      f"ER(W_in)={hist['W_in_erank'][-1]:.1f}", flush=True)
        if step in keep:
            ckpts[step] = plain_state(model, c["factor"])
    ckpts[last] = plain_state(model, c["factor"])
    return {"seed": seed, "config": c, "hist": {k: np.asarray(v) for k, v in hist.items()},
            "ckpts": ckpts, "final_step": last, "train_idx": tr.cpu(), "test_idx": te.cpu()}


def parse_seeds(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-14")
    ap.add_argument("--wd", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--ckpt-every", type=int, default=None, help="extra checkpoint every N steps")
    ap.add_argument("--log-every", type=int, default=None)
    ap.add_argument("--factor", default=None, help="MATRIX:RANK, e.g. W_in:16 or W_E:128")
    ap.add_argument("--out", default="results")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    cfg = {"wd": args.wd}
    if args.steps:
        cfg["steps"] = args.steps
    if args.ckpt_every:
        cfg["ckpt_every"] = args.ckpt_every
    if args.log_every:
        cfg["log_every"] = args.log_every
    sub = os.path.join("transformer", f"wd{args.wd:g}")
    if args.factor:
        name, r = args.factor.split(":")
        if name not in FACTOR_TARGETS:
            ap.error(f"--factor supports {sorted(FACTOR_TARGETS)}")
        cfg["factor"] = (name, int(r))
        sub = os.path.join("transformer_cap", f"{name}_r{int(r)}", f"wd{args.wd:g}")
    for seed in parse_seeds(args.seeds):
        path = os.path.join(args.out, sub, f"seed{seed}.pt")
        if os.path.exists(path) and not args.overwrite:
            print(f"[skip] {path}")
            continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        t0 = time.time()
        print(f"[run] transformer seed={seed} wd={args.wd}", flush=True)
        run = train_run(seed, cfg, args.device)
        torch.save(run, path)
        print(f"[done] {path}  final test_acc={run['hist']['test_acc'][-1]:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
