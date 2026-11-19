"""XGBoost baselines with the same two-stage protocol as the GNNs."""

from __future__ import annotations

import os

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


def train_xgb(
    x: np.ndarray,
    y: np.ndarray,
    masks: dict[str, np.ndarray],
    seed: int = 0,
    max_rounds: int = 1000,
    early_stopping: int = 50,
    params: dict | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Early-stop on the validation block, then refit on the full window with that many trees.

    Returns (validation-stage probabilities, final probabilities, chosen rounds) for all rows.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    n_jobs = min(8, os.cpu_count() or 1)
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
    final = XGBClassifier(n_estimators=rounds, random_state=seed, n_jobs=n_jobs, **p)
    final.fit(x[train], y[train], verbose=False)
    return p_val_stage, final.predict_proba(x)[:, 1], rounds
