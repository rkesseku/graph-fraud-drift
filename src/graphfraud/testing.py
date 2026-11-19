"""Synthetic Elliptic-format data for tests and demos (no download needed)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import CLASSES_FILE, EDGES_FILE, FEATURES_FILE


def write_synthetic_elliptic(
    raw_dir: str | Path,
    n_steps: int = 12,
    nodes_per_step: int = 120,
    n_local: int = 6,
    n_agg: int = 4,
    illicit_rate: float = 0.15,
    unknown_rate: float = 0.4,
    shift_step: int | None = None,
    seed: int = 0,
) -> Path:
    """Write the three Elliptic CSVs with a learnable signal.

    Illicit nodes have shifted local features and preferentially link to other illicit nodes.
    From ``shift_step`` on, the illicit feature shift changes direction (a toy "drift").
    """
    rng = np.random.default_rng(seed)
    raw = Path(raw_dir)
    raw.mkdir(parents=True, exist_ok=True)
    rows, classes, edges = [], [], []
    next_id = 1000
    for s in range(1, n_steps + 1):
        ids = np.arange(next_id, next_id + nodes_per_step)
        next_id += nodes_per_step
        illicit = rng.random(nodes_per_step) < illicit_rate
        direction = -1.0 if shift_step is not None and s >= shift_step else 1.0
        x = rng.normal(size=(nodes_per_step, n_local + n_agg))
        x[illicit, :n_local] += direction * 1.5
        for i in range(nodes_per_step):
            same = np.flatnonzero(illicit == illicit[i])
            for j in rng.choice(same, size=2):
                if j != i:
                    edges.append((ids[i], ids[j]))
            if rng.random() < 0.3:
                edges.append((ids[i], ids[rng.integers(nodes_per_step)]))
        unknown = rng.random(nodes_per_step) < unknown_rate
        for i in range(nodes_per_step):
            rows.append([ids[i], s, *x[i]])
            label = "unknown" if unknown[i] else ("1" if illicit[i] else "2")
            classes.append((ids[i], label))
    pd.DataFrame(rows).to_csv(raw / FEATURES_FILE, header=False, index=False)
    pd.DataFrame(classes, columns=["txId", "class"]).to_csv(raw / CLASSES_FILE, index=False)
    e = pd.DataFrame(edges, columns=["txId1", "txId2"]).drop_duplicates()
    e = e[e.txId1 != e.txId2]
    e.to_csv(raw / EDGES_FILE, index=False)
    return raw
