"""
common_mlp.py -- shared code for the mean-field MLP experiments.

Default configuration = the verified 15-seed run:
  MSE loss on one-hot targets, quadratic activation, width 128,
  mean-field scaling (pre-activations / sqrt(2p), output / width),
  AdamW(lr=1e-3, weight_decay=1e-2), 35% train split, full batch,
  100k steps, metrics logged every 500 steps.

The random-number order (seed -> train/test split -> W1 -> W2) and the
metric definitions (effective rank, Fourier IPR, onset rules).
"""
import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

P = 97
WIDTH = 128
EPS = 1e-10

DEFAULTS = dict(
    p=P, width=WIDTH, alpha=0.35, steps=100_000, lr=1e-3, wd=1e-2,
    log_every=500, loss="mse", act="quadratic", rank=None,
    lam_nuc=0.0, nuc_start=0, labels="true", op="add",
    optimizer="adamw",            # "adamw" or "sgd" (same per-step decay lr * wd * w)
    eps=1e-8, momentum=0.0,       # AdamW epsilon and SGD momentum (torch defaults)
    init_ckpt=None, init_step=0,  # continue from checkpoint `init_step` of a saved run
)

# Checkpoints kept for the stability / transplant analyses. Steps beyond the
# run length are skipped; the final weights are always stored.
CKPT_STEPS = [0, 2500, 5000, 7500, 8000, 9000, 10000, 11000, 12500,
              15000, 20000, 30000, 50000]

HIST_KEYS = ["steps", "train_loss", "test_loss", "train_acc", "test_acc",
             "W1_erank", "W2_erank", "W1_srank", "W2_srank",
             "fourier_ipr", "spectral_ipr", "fourier_erank",
             "W1_hoyer", "W1_l0", "W1_norm", "W2_norm", "grad_norm",
             "radial_update", "radial_decay"]


# ---------------------------------------------------------------- data
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_dataset(p=P, op="add"):
    """Rows ordered (a, b) with a outer, b inner -- same order as NEW.py."""
    a = torch.arange(p).repeat_interleave(p)
    b = torch.arange(p).repeat(p)
    X = torch.zeros(p * p, 2 * p)
    rows = torch.arange(p * p)
    X[rows, a] = 1.0
    X[rows, p + b] = 1.0
    y = {"add": a + b, "sub": a - b, "mul": a * b}[op] % p
    return X, y


# ---------------------------------------------------------------- model
class MeanFieldMLP(nn.Module):
    """h = W1 x / sqrt(2p),  z = act(h),  f = W2 z / width.

    rank=None trains W1 directly; rank=r trains W1 = A @ B (a hard rank-r
    bottleneck from initialisation).
    """

    def __init__(self, p=P, width=WIDTH, act="quadratic", rank=None):
        super().__init__()
        self.p, self.width, self.act_name, self.rank = p, width, act, rank
        if rank is None:
            self.W1_full = nn.Parameter(torch.randn(width, 2 * p))
        else:
            s = rank ** -0.25          # keeps Var[(AB)_ij] = 1, like randn
            self.A = nn.Parameter(torch.randn(width, rank) * s)
            self.B = nn.Parameter(torch.randn(rank, 2 * p) * s)
        self.W2 = nn.Parameter(torch.randn(p, width))

    @property
    def W1(self):
        return self.W1_full if self.rank is None else self.A @ self.B

    def pre(self, x):
        return (self.W1 @ x.T).T / np.sqrt(x.shape[1])

    def act(self, h):
        if self.act_name == "quadratic":
            return h ** 2
        if self.act_name == "relu":
            return F.relu(h)
        if self.act_name == "gelu":
            return F.gelu(h)
        raise ValueError(self.act_name)

    def readout(self, z):
        return (self.W2 @ z.T).T / self.width

    def forward(self, x):
        return self.readout(self.act(self.pre(x)))


def model_from_weights(W1, W2, act="quadratic"):
    """Full-rank model carrying the given weights (used for analysis)."""
    m = MeanFieldMLP(W1.shape[1] // 2, W1.shape[0], act)
    with torch.no_grad():
        m.W1_full.copy_(W1)
        m.W2.copy_(W2)
    return m


def compute_loss(out, y, loss):
    if loss == "mse":
        return F.mse_loss(out, F.one_hot(y, out.shape[1]).float())
    if loss == "ce":
        return F.cross_entropy(out, y)
    raise ValueError(loss)


def accuracy(out, y):
    return float((out.argmax(1) == y).float().mean())


# ---------------------------------------------------------------- metrics
def _sv(W):
    return torch.linalg.svdvals(W.detach().float())


def effective_rank(W):
    s = _sv(W)
    q = s / (s.sum() + EPS)
    return float(torch.exp(-(q * torch.log(q + EPS)).sum()))


def stable_rank(W):
    s = _sv(W)
    return float((s ** 2).sum() / (s[0] ** 2 + EPS))


def spectral_ipr(W):
    """sum_i (sigma_i / sum_j sigma_j)^4 -- the notebook's spectral IPR."""
    s = _sv(W)
    q = s / (s.sum() + EPS)
    return float((q ** 4).sum())


def fourier_ipr(W1, p=P):
    """Doshi et al. IPR of the DFT of each unit's weights on operand a,
    averaged over units (identical to NEW.py)."""
    Wf = np.fft.fft(W1.detach().cpu().numpy()[:, :p], axis=1)
    A = np.abs(Wf)
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + EPS)
    return float(np.mean(np.sum(A ** 4, axis=1)))


def fourier_effective_rank(W1, p=P):
    Wf = torch.fft.fft(W1.detach().float()[:, :p], dim=1)
    M = torch.view_as_real(Wf).reshape(W1.shape[0], -1)
    return effective_rank(M)


def hoyer_sparsity(W):
    w = W.detach().flatten().abs()
    n = w.numel()
    return float((math.sqrt(n) - w.sum() / (w.norm() + EPS)) / (math.sqrt(n) - 1))


def l0_sparsity(W, thr=1e-3):
    return float((W.detach().abs() < thr).float().mean())


# ---------------------------------------------------------------- training
def make_optimizer(model, c):
    """AdamW (decoupled decay) or plain SGD with weight decay. Both shrink every
    weight by lr * wd * w per step; they differ only in the loss-driven step."""
    c = {**DEFAULTS, **c}              # configs saved before eps / momentum existed
    if c["optimizer"] == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=c["lr"], weight_decay=c["wd"], eps=c["eps"])
    if c["optimizer"] == "sgd":
        return torch.optim.SGD(model.parameters(), lr=c["lr"], weight_decay=c["wd"],
                               momentum=c["momentum"])
    raise ValueError(c["optimizer"])


def train_run(seed, cfg=None, device="cpu", ckpt_steps=CKPT_STEPS, verbose=True):
    """Train one model; return a dict with config, history, checkpoints."""
    c = {**DEFAULTS, **(cfg or {})}
    set_seed(seed)
    X, y = make_dataset(c["p"], c["op"])
    perm = torch.randperm(len(X))
    n_tr = int(c["alpha"] * len(X))
    tr, te = perm[:n_tr], perm[n_tr:]
    if c["labels"] == "random":
        g = torch.Generator().manual_seed(10_000 + seed)
        y = y[torch.randperm(len(y), generator=g)]

    model = MeanFieldMLP(c["p"], c["width"], c["act"], c["rank"]).to(device)
    off = 0
    if c["init_ckpt"]:                 # continue a saved run: same seed, same split
        if c["rank"] is not None:
            raise ValueError("init_ckpt supports the unfactorised model only")
        base = load_run(c["init_ckpt"])
        if not torch.equal(base["train_idx"], tr.cpu()):
            raise ValueError(f"{c['init_ckpt']} has a different train split; use the same seed")
        w = base["ckpts"][c["init_step"]]
        with torch.no_grad():
            model.W1_full.copy_(w["W1"])
            model.W2.copy_(w["W2"])
        off = c["init_step"]
    X, y = X.to(device), y.to(device)
    Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
    opt = make_optimizer(model, c)
    params = list(model.parameters())

    hist = {k: [] for k in HIST_KEYS}
    ckpts = {}
    keep = {s for s in ckpt_steps if off <= s < off + c["steps"]}
    last = off + c["steps"] - 1             # steps are counted from the start of the base run

    for i in range(c["steps"]):
        step = off + i
        opt.zero_grad()
        out = model(Xtr)
        loss = compute_loss(out, ytr, c["loss"])
        total = loss
        if c["lam_nuc"] > 0 and step >= c["nuc_start"]:
            total = loss + c["lam_nuc"] * torch.linalg.matrix_norm(model.W1, ord="nuc")
        total.backward()
        log_now = (step % c["log_every"] == 0) or step == last
        if log_now:
            gnorm = math.sqrt(sum(float(q.grad.pow(2).sum())
                                  for q in model.parameters() if q.grad is not None))
            old = [q.detach().clone() for q in params]
        opt.step()
        if log_now:
            # radial forces on the weight norm in this step: <theta, delta theta> in total,
            # and the part due to weight decay (-lr * wd * ||theta||^2); the rest is the
            # loss-driven (optimizer) part
            with torch.no_grad():
                r_upd = sum(float((o * (q - o)).sum()) for o, q in zip(old, params))
                r_dec = -c["lr"] * c["wd"] * sum(float(o.pow(2).sum()) for o in old)

        if log_now:
            with torch.no_grad():
                W1, W2 = model.W1, model.W2
                o_tr, o_te = model(Xtr), model(Xte)
                hist["steps"].append(step)
                hist["train_loss"].append(float(compute_loss(o_tr, ytr, c["loss"])))
                hist["test_loss"].append(float(compute_loss(o_te, yte, c["loss"])))
                hist["train_acc"].append(accuracy(o_tr, ytr))
                hist["test_acc"].append(accuracy(o_te, yte))
                hist["W1_erank"].append(effective_rank(W1))
                hist["W2_erank"].append(effective_rank(W2))
                hist["W1_srank"].append(stable_rank(W1))
                hist["W2_srank"].append(stable_rank(W2))
                hist["fourier_ipr"].append(fourier_ipr(W1, c["p"]))
                hist["spectral_ipr"].append(spectral_ipr(W1))
                hist["fourier_erank"].append(fourier_effective_rank(W1, c["p"]))
                hist["W1_hoyer"].append(hoyer_sparsity(W1))
                hist["W1_l0"].append(l0_sparsity(W1))
                hist["W1_norm"].append(float(W1.norm()))
                hist["W2_norm"].append(float(W2.norm()))
                hist["grad_norm"].append(gnorm)
                hist["radial_update"].append(r_upd)
                hist["radial_decay"].append(r_dec)
            if verbose and step % (c["log_every"] * 20) == 0:
                print(f"    step {step:>6}  train_acc={hist['train_acc'][-1]:.3f}  "
                      f"test_acc={hist['test_acc'][-1]:.3f}  ER(W1)={hist['W1_erank'][-1]:.1f}",
                      flush=True)

        if step in keep:
            ckpts[step] = {"W1": model.W1.detach().cpu().clone(),
                           "W2": model.W2.detach().cpu().clone()}

    ckpts[last] = {"W1": model.W1.detach().cpu().clone(),
                   "W2": model.W2.detach().cpu().clone()}
    return {
        "seed": seed, "config": c,
        "hist": {k: np.asarray(v) for k, v in hist.items()},
        "ckpts": ckpts, "final_step": last,
        "train_idx": tr.cpu(), "test_idx": te.cpu(), "labels": y.cpu(),
    }


def load_run(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def run_data(run, device="cpu"):
    """Rebuild (X, y, train_idx, test_idx) for a saved run."""
    c = run["config"]
    X, _ = make_dataset(c["p"], c["op"])
    return X.to(device), run["labels"].to(device), run["train_idx"], run["test_idx"]


def final_model(run, device="cpu"):
    w = run["ckpts"][run["final_step"]]
    return model_from_weights(w["W1"], w["W2"], run["config"]["act"]).to(device)


# ---------------------------------------------------------------- onsets
def grokking_onset(steps, test_acc, thr=0.95):
    """First logged step with test accuracy >= thr (NEW.py rule)."""
    for s, a in zip(steps, test_acc):
        if a >= thr:
            return int(s)
    return None


def compression_onset(steps, erank, min_step=1000, drop=0.01):
    """First step >= min_step where ER falls more than `drop` below its
    running maximum (NEW.py rule)."""
    running = erank[0]
    for s, e in zip(steps, erank):
        running = max(running, e)
        if s < min_step:
            continue
        if e < (1 - drop) * running:
            return int(s)
    return None


def detect_onset(steps, vals, window_end=5000, k=3.0, direction="up"):
    """First step at or after `window_end` where `vals` leaves the baseline
    band mean +/- k*std computed over steps < window_end.

    direction: 'up' (value > mean + k sd), 'down' (< mean - k sd) or 'both'.
    With window_end=5000, k=3, direction='up' this equals NEW.py's IPR onset.
    """
    steps, vals = np.asarray(steps), np.asarray(vals, dtype=float)
    base = vals[steps < window_end]
    if len(base) < 2:
        return None
    mu, sd = base.mean(), base.std() + EPS
    for s, v in zip(steps, vals):
        if s < window_end:
            continue
        if (direction in ("up", "both") and v > mu + k * sd) or \
           (direction in ("down", "both") and v < mu - k * sd):
            return int(s)
    return None
