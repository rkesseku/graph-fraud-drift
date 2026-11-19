"""Loading the Elliptic Bitcoin transaction graph (Weber et al., 2019).

The release ships three CSV files:

* ``elliptic_txs_features.csv`` (no header): ``txId``, ``time step``, 93 local features,
  72 aggregated (one-hop neighbourhood) features.
* ``elliptic_txs_classes.csv``: ``txId,class`` with ``1`` = illicit, ``2`` = licit, ``unknown``.
* ``elliptic_txs_edgelist.csv``: ``txId1,txId2`` directed payment flows.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

N_LOCAL = 93
N_AGG = 72
FEATURES_FILE = "elliptic_txs_features.csv"
CLASSES_FILE = "elliptic_txs_classes.csv"
EDGES_FILE = "elliptic_txs_edgelist.csv"

ILLICIT, LICIT, UNKNOWN = 1, 0, -1


@dataclass
class EllipticGraph:
    """In-memory graph. Node ``i`` is row ``i`` of every per-node array."""

    tx_id: np.ndarray  # (N,) int64
    step: np.ndarray  # (N,) int64, 1-based time step
    x: np.ndarray  # (N, n_local + n_agg) float32; the time step itself is NOT a feature
    y: np.ndarray  # (N,) int8: 1 illicit, 0 licit, -1 unknown
    edge_index: np.ndarray  # (2, E) int64 node indices, directed src -> dst
    n_local: int = N_LOCAL

    @property
    def num_nodes(self) -> int:
        return int(self.x.shape[0])

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])

    @property
    def x_local(self) -> np.ndarray:
        return self.x[:, : self.n_local]

    @property
    def labelled(self) -> np.ndarray:
        return self.y >= 0

    def summary(self) -> dict:
        lab = self.labelled
        n_illicit = int((self.y == ILLICIT).sum())
        return {
            "nodes": self.num_nodes,
            "edges": self.num_edges,
            "time_steps": int(self.step.max()),
            "features_local": self.n_local,
            "features_aggregated": int(self.x.shape[1] - self.n_local),
            "labelled": int(lab.sum()),
            "illicit": n_illicit,
            "licit": int((self.y == LICIT).sum()),
            "unknown": int((~lab).sum()),
            "illicit_share_of_labelled": n_illicit / max(int(lab.sum()), 1),
        }


def _map_class(values: pd.Series) -> np.ndarray:
    mapping = {"1": ILLICIT, "2": LICIT, "unknown": UNKNOWN}
    out = values.astype(str).map(mapping)
    if out.isna().any():
        bad = sorted(values[out.isna()].astype(str).unique())[:5]
        raise ValueError(f"unexpected class labels: {bad}")
    return out.to_numpy(dtype=np.int8)


def load_elliptic(raw_dir: str | Path, n_local: int = N_LOCAL) -> EllipticGraph:
    """Read the three CSV files from ``raw_dir`` into an :class:`EllipticGraph`."""
    raw = Path(raw_dir)
    for name in (FEATURES_FILE, CLASSES_FILE, EDGES_FILE):
        if not (raw / name).exists():
            raise FileNotFoundError(f"missing {raw / name}; see README for download instructions")

    feats = pd.read_csv(raw / FEATURES_FILE, header=None)
    n_feat = feats.shape[1] - 2
    if n_feat < 1:
        raise ValueError("features file must have txId, time step and at least one feature")
    tx_id = feats[0].to_numpy(dtype=np.int64)
    step = feats[1].to_numpy(dtype=np.int64)
    x = feats.iloc[:, 2:].to_numpy(dtype=np.float32)

    classes = pd.read_csv(raw / CLASSES_FILE)
    classes.columns = ["txId", "class"]
    cls = pd.Series(_map_class(classes["class"]), index=classes["txId"].to_numpy(dtype=np.int64))
    y = cls.reindex(tx_id).fillna(UNKNOWN).to_numpy(dtype=np.int8)

    edges = pd.read_csv(raw / EDGES_FILE)
    edges.columns = ["src", "dst"]
    index = pd.Series(np.arange(len(tx_id), dtype=np.int64), index=tx_id)
    src = index.reindex(edges["src"].to_numpy(dtype=np.int64)).to_numpy(dtype=float)
    dst = index.reindex(edges["dst"].to_numpy(dtype=np.int64)).to_numpy(dtype=float)
    keep = ~(np.isnan(src) | np.isnan(dst))
    edge_index = np.vstack([src[keep], dst[keep]]).astype(np.int64)
    return EllipticGraph(
        tx_id=tx_id, step=step, x=x, y=y, edge_index=edge_index, n_local=min(n_local, n_feat)
    )
