"""XGBoost baselines with the same two-stage protocol as the GNNs."""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
from xgboost import XGBClassifier

DEFAULT_PARAMS = {
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 1.0,
    "tree_method": "hist",
    "eval_metric": "aucpr",
}


class TwoStageResult(NamedTuple):
    """Scores for all rows from the two-stage protocol.

    ``p_stage1``: model fitted on the fit block (early-stopped on validation).
    ``p_final``: the deployed model, refit on the full training window with the chosen size.
    ``size``: chosen tree (or epoch) count.
    ``p_refits``: one score vector per extra training window, each from a model refit with the
    *same recipe* as the final model (same size, seed and hyperparameters).
    """

    p_stage1: np.ndarray
    p_final: np.ndarray
    size: int
    p_refits: list[np.ndarray]


def default_n_jobs() -> int:
    return min(8, os.cpu_count() or 1)


def train_xgb(
    x: np.ndarray,
    y: np.ndarray,
    masks: dict[str, np.ndarray],
    seed: int = 0,
    max_rounds: int = 1000,
    early_stopping: int = 50,
    params: dict | None = None,
    refit_masks: Sequence[np.ndarray] = (),
    n_jobs: int | None = None,
) -> TwoStageResult:
    """Early-stop on the validation block, then refit on the full window with that many trees.

    ``refit_masks`` are extra training windows on which the final recipe is refit (used for
    rolling-origin threshold selection).
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    n_jobs = n_jobs or default_n_jobs()
    fit, val, train = masks["fit"], masks["val"], masks["train"]
    stage1 = XGBClassifier(
        n_estimators=max_rounds,
        early_stopping_rounds=early_stopping,
        random_state=seed,
        n_jobs=n_jobs,
        **p,
    )
    stage1.fit(x[fit], y[fit], eval_set=[(x[val], y[val])], verbose=False)
    rounds = int(stage1.best_iteration) + 1
    p_val_stage = stage1.predict_proba(x)[:, 1]

    def refit(mask: np.ndarray) -> np.ndarray:
        model = XGBClassifier(n_estimators=rounds, random_state=seed, n_jobs=n_jobs, **p)
        model.fit(x[mask], y[mask], verbose=False)
        return model.predict_proba(x)[:, 1]

    return TwoStageResult(p_val_stage, refit(train), rounds, [refit(m) for m in refit_masks])
