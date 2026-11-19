"""Illicit-class metrics, cost-sensitive thresholds and a time-step block bootstrap."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def confusion(y: np.ndarray, pred: np.ndarray) -> tuple[int, int, int, int]:
    """(TP, FP, FN, TN) with illicit = 1 as the positive class."""
    y = y.astype(bool)
    pred = pred.astype(bool)
    return (
        int((y & pred).sum()),
        int((~y & pred).sum()),
        int((y & ~pred).sum()),
        int((~y & ~pred).sum()),
    )


def prf(tp: float, fp: float, fn: float) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def illicit_metrics(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict:
    tp, fp, fn, tn = confusion(y, p >= threshold)
    precision, recall, f1 = prf(tp, fp, fn)
    both = len(np.unique(y)) == 2
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auprc": float(average_precision_score(y, p)) if both else float("nan"),
        "auroc": float(roc_auc_score(y, p)) if both else float("nan"),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "threshold": float(threshold),
    }


def expected_cost(
    y: np.ndarray, p: np.ndarray, threshold: float, c_fn: float, c_fp: float
) -> float:
    """Average cost per transaction: c_fn per missed illicit, c_fp per false alert."""
    tp, fp, fn, _ = confusion(y, p >= threshold)
    return (c_fn * fn + c_fp * fp) / max(len(y), 1)


def candidate_thresholds(p: np.ndarray, n: int = 400) -> np.ndarray:
    qs = np.unique(np.quantile(p, np.linspace(0.0, 1.0, n)))
    return np.unique(np.concatenate([qs, [0.5]]))


def cost_optimal_threshold(y: np.ndarray, p: np.ndarray, c_fn: float, c_fp: float) -> float:
    """Threshold minimising expected cost on (y, p); ties go to the higher threshold."""
    ts = candidate_thresholds(p)
    costs = np.array([expected_cost(y, p, t, c_fn, c_fp) for t in ts])
    best = np.flatnonzero(costs == costs.min())
    return float(ts[best[-1]])


def counts_by_step(y: np.ndarray, pred: np.ndarray, step: np.ndarray) -> tuple[np.ndarray, dict]:
    """Per-step (TP, FP, FN) counts, rows ordered by step."""
    steps = np.unique(step)
    rows = []
    for s in steps:
        m = step == s
        tp, fp, fn, _ = confusion(y[m], pred[m])
        rows.append((tp, fp, fn))
    return steps, np.array(rows, dtype=float)


def per_step_f1(y: np.ndarray, p: np.ndarray, step: np.ndarray, threshold: float) -> dict:
    steps, c = counts_by_step(y, p >= threshold, step)
    return {int(s): prf(*row)[2] for s, row in zip(steps, c, strict=True)}


def block_bootstrap_f1(
    counts: dict[str, list[np.ndarray]],
    reference: str,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, dict]:
    """Bootstrap over *time steps* (the natural unit of dependence).

    ``counts[model]`` is a list (one per seed) of (S, 3) per-step TP/FP/FN arrays over the same
    steps. Each replicate resamples steps with replacement, pools counts, computes F1 per seed
    and averages over seeds. Returns the mean-seed F1 interval for every model and the interval
    of its difference from ``reference``.
    """
    rng = np.random.default_rng(seed)
    n_steps = counts[reference][0].shape[0]
    idx = rng.integers(0, n_steps, size=(n_boot, n_steps))

    def f1_reps(arrs: list[np.ndarray]) -> np.ndarray:
        reps = []
        for a in arrs:
            pooled = a[idx].sum(axis=1)  # (B, 3)
            tp, fp, fn = pooled.T
            denom = 2 * tp + fp + fn
            reps.append(np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0))
        return np.mean(reps, axis=0)

    ref = f1_reps(counts[reference])
    out = {}
    for name, arrs in counts.items():
        reps = f1_reps(arrs)
        diff = reps - ref
        out[name] = {
            "f1_ci95": [float(np.quantile(reps, 0.025)), float(np.quantile(reps, 0.975))],
            "diff_vs_reference": float(diff.mean()),
            "diff_ci95": [float(np.quantile(diff, 0.025)), float(np.quantile(diff, 0.975))],
        }
    return out


def block_bootstrap_paired(
    y: np.ndarray,
    step: np.ndarray,
    arms: dict[str, list[tuple[np.ndarray, float]]],
    reference: str,
    metric: Callable[[np.ndarray, np.ndarray, float], float],
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, dict]:
    """Paired time-step bootstrap of any row-level metric (AUPRC, cost, ...).

    ``arms[name]`` holds one ``(scores, threshold)`` pair per seed, all scored on the same rows
    ``(y, step)``. Each replicate resamples time steps with replacement (the same draw for every
    arm, so differences are paired), evaluates ``metric(y, scores, threshold)`` per seed on the
    resampled rows and averages over seeds. Uses the same step draw as
    :func:`block_bootstrap_f1` for equal ``seed`` and number of steps.
    """
    rng = np.random.default_rng(seed)
    steps = np.unique(step)
    idx = rng.integers(0, len(steps), size=(n_boot, len(steps)))
    rows = [np.flatnonzero(step == s) for s in steps]
    reps = {name: np.empty(n_boot) for name in arms}
    for b in range(n_boot):
        sel = np.concatenate([rows[i] for i in idx[b]])
        ys = y[sel]
        for name, pairs in arms.items():
            reps[name][b] = np.mean([metric(ys, p[sel], t) for p, t in pairs])
    point = {
        name: float(np.mean([metric(y, p, t) for p, t in pairs])) for name, pairs in arms.items()
    }
    out = {}
    for name in arms:
        diff = reps[name] - reps[reference]
        out[name] = {
            "estimate": point[name],
            "ci95": [float(np.quantile(reps[name], 0.025)), float(np.quantile(reps[name], 0.975))],
            "diff_vs_reference": point[name] - point[reference],
            "diff_ci95": [float(np.quantile(diff, 0.025)), float(np.quantile(diff, 0.975))],
        }
    return out
