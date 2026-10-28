"""Metrics for the benchmark harness. Pure numpy (sklearn only for macro-F1)."""
from __future__ import annotations

import numpy as np


def macro_f1(y_true, y_pred, labels=(-1, 0, 1)) -> float:
    from sklearn.metrics import f1_score
    return float(f1_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0))


def accuracy(y_true, y_pred) -> float:
    a, b = np.asarray(y_true), np.asarray(y_pred)
    return float((a == b).mean()) if len(a) else 0.0


def mae(y_true, y_pred) -> float:
    a, b = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    return float(np.abs(a - b).mean()) if len(a) else 0.0


def ece(probs: np.ndarray, y_true_idx: np.ndarray, bins: int = 15) -> float:
    """Top-label expected calibration error, equal-width bins on confidence."""
    probs = np.asarray(probs, dtype=float)
    y = np.asarray(y_true_idx)
    n = len(y)
    if n == 0:
        return 0.0
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y).astype(float)
    idx = np.minimum((conf * bins).astype(int), bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.sum() / n * abs(correct[m].mean() - conf[m].mean())
    return float(total)


def to3(probs5: np.ndarray) -> np.ndarray:
    p = np.asarray(probs5, dtype=float)
    return np.stack([p[:, 0] + p[:, 1], p[:, 2], p[:, 3] + p[:, 4]], axis=1)


def confusion(y_true, y_pred, labels) -> list[list[int]]:
    pos = {l: i for i, l in enumerate(labels)}
    m = [[0] * len(labels) for _ in labels]
    for t, p in zip(y_true, y_pred):
        if t in pos and p in pos:
            m[pos[t]][pos[p]] += 1
    return m


# --- 5-class probabilistic scores. p5: (n, 5) over classes -2..2; y: ints in -2..2. ---------------

def _p5_y(p5, y) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(p5, dtype=float).reshape(-1, 5)
    t = np.asarray(y, dtype=int).reshape(-1)
    if len(p) != len(t):
        raise ValueError(f"p5 has {len(p)} rows, y has {len(t)}")
    return p, t + 2


def rps(p5, y) -> float:
    """Ranked probability score, mean over rows: sum of squared CDF gaps over the 4 inner cut
    points, divided by 4 (0 = perfect, 1 = all mass on the far end)."""
    p, t = _p5_y(p5, y)
    if not len(t):
        return 0.0
    onehot = np.eye(5)[t]
    gap = np.cumsum(p, axis=1)[:, :4] - np.cumsum(onehot, axis=1)[:, :4]
    return float((gap ** 2).sum(axis=1).mean() / 4)


def log_loss5(p5, y) -> float:
    p, t = _p5_y(p5, y)
    if not len(t):
        return 0.0
    return float(-np.log(np.clip(p[np.arange(len(t)), t], 1e-12, 1.0)).mean())


def brier5(p5, y) -> float:
    """Multi-class Brier: mean over rows of the sum of squared errors against the one-hot target."""
    p, t = _p5_y(p5, y)
    if not len(t):
        return 0.0
    return float(((p - np.eye(5)[t]) ** 2).sum(axis=1).mean())


def ece_eqmass(p5, y, bins: int = 10) -> float:
    """Top-label ECE with equal-count bins (rows sorted by confidence, stable on ties)."""
    p, t = _p5_y(p5, y)
    n = len(t)
    if n == 0:
        return 0.0
    conf = p.max(axis=1)
    correct = (p.argmax(axis=1) == t).astype(float)
    order = np.argsort(conf, kind="stable")
    total = 0.0
    for idx in np.array_split(order, min(bins, n)):
        total += len(idx) / n * abs(correct[idx].mean() - conf[idx].mean())
    return float(total)


def ece_floor(p5, bins: int = 10, n_sim: int = 500, seed: int = 0) -> float:
    """Mean ece_eqmass when the labels are drawn from p5 itself: the ECE a perfectly calibrated
    model still shows at this n and bin count. Read ECE relative to it."""
    p = np.asarray(p5, dtype=float).reshape(-1, 5)
    if not len(p):
        return 0.0
    rng = np.random.default_rng(seed)
    cdf = np.cumsum(p / p.sum(axis=1, keepdims=True), axis=1)
    cdf[:, -1] = 1.0
    sims = []
    for _ in range(n_sim):
        u = rng.random((len(p), 1))
        sims.append(ece_eqmass(p, (u > cdf).sum(axis=1) - 2, bins))
    return float(np.mean(sims))


# --- uncertainty -----------------------------------------------------------------------------

def _rows(arrays) -> tuple[list[np.ndarray], int]:
    arrs = [np.asarray(a) for a in arrays]
    lengths = {len(a) for a in arrs}
    if len(lengths) != 1:
        raise ValueError(f"arrays differ in length: {sorted(lengths)}")
    return arrs, lengths.pop()


def bootstrap_ci(fn, *arrays, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% percentile CI of fn(*arrays) under row resampling (same indices for every array)."""
    arrs, size = _rows(arrays)
    if size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    stats = [fn(*(a[idx] for a in arrs)) for idx in rng.integers(0, size, (n, size))]
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def paired_bootstrap(fn, a_args, b_args, n: int = 2000, seed: int = 0) -> dict:
    """fn(*a_args) - fn(*b_args) with a 95% CI; both sides use the same resampled rows."""
    arrs, size = _rows([*a_args, *b_args])
    a, b = arrs[:len(a_args)], arrs[len(a_args):]
    diff = float(fn(*a) - fn(*b))
    if size == 0:
        return {"diff": diff, "lo": float("nan"), "hi": float("nan")}
    rng = np.random.default_rng(seed)
    stats = [fn(*(x[idx] for x in a)) - fn(*(x[idx] for x in b)) for idx in rng.integers(0, size, (n, size))]
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return {"diff": diff, "lo": float(lo), "hi": float(hi)}


def mcnemar(correct_a, correct_b) -> float:
    """Exact two-sided McNemar p-value: binomial(0.5) test on the discordant pairs."""
    from math import comb
    a, b = np.asarray(correct_a, dtype=bool), np.asarray(correct_b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("correct_a and correct_b differ in shape")
    only_a, only_b = int((a & ~b).sum()), int((~a & b).sum())
    m = only_a + only_b
    if m == 0:
        return 1.0
    tail = sum(comb(m, i) for i in range(min(only_a, only_b) + 1)) / 2 ** m
    return float(min(1.0, 2 * tail))
