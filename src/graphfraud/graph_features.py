"""Label-free structural features.

Nothing here reads labels, so computing these over the whole graph (test steps included) leaks
no target information. Elliptic edges only connect transactions within one time step, so a test
node's features depend only on its own step's subgraph.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def degrees(edge_index: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (in-degree, out-degree)."""
    src, dst = edge_index
    return np.bincount(dst, minlength=n), np.bincount(src, minlength=n)


def symmetric_adjacency(edge_index: np.ndarray, n: int) -> sp.csr_matrix:
    """Binary undirected adjacency without self-loops (duplicate edges collapsed)."""
    src, dst = edge_index
    a = sp.coo_matrix((np.ones(len(src)), (src, dst)), shape=(n, n)).tocsr()
    a = ((a + a.T) > 0).astype(np.float64).tolil()
    a.setdiag(0)
    a = a.tocsr()
    a.eliminate_zeros()
    return a


def pagerank(
    edge_index: np.ndarray, n: int, alpha: float = 0.85, max_iter: int = 200, tol: float = 1e-12
) -> np.ndarray:
    """Directed PageRank by power iteration; dangling mass is spread uniformly. Sums to 1."""
    src, dst = edge_index
    out_deg = np.bincount(src, minlength=n).astype(np.float64)
    w = 1.0 / out_deg[src]
    m = sp.csr_matrix((w, (dst, src)), shape=(n, n))
    dangling = out_deg == 0
    r = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        r_new = alpha * (m @ r + r[dangling].sum() / n) + (1.0 - alpha) / n
        done = np.abs(r_new - r).sum() < tol
        r = r_new
        if done:
            break
    return r / r.sum()


def neighbour_mean(adj: sp.csr_matrix, x: np.ndarray) -> np.ndarray:
    """Row-normalised neighbourhood mean; isolated nodes get zeros."""
    deg = np.asarray(adj.sum(axis=1)).ravel()
    inv = np.divide(1.0, deg, out=np.zeros_like(deg), where=deg > 0)
    return np.asarray(sp.diags(inv) @ (adj @ x), dtype=np.float32)


def graph_features(edge_index: np.ndarray, x_local: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Return (N, 5 + 2 * F_local) structural features and their names.

    * log in-, out- and total degree, log of PageRank scaled by N, log mean neighbour degree;
    * one-hop and two-hop neighbourhood means of the local features. Elliptic's own 72
      aggregated features already summarise the one-hop neighbourhood; the two-hop means and
      the structural scores add information beyond them.
    """
    n = x_local.shape[0]
    indeg, outdeg = degrees(edge_index, n)
    adj = symmetric_adjacency(edge_index, n)
    und_deg = np.asarray(adj.sum(axis=1)).ravel()
    nbr_deg = neighbour_mean(adj, und_deg[:, None]).ravel()
    pr = pagerank(edge_index, n)
    hop1 = neighbour_mean(adj, x_local)
    hop2 = neighbour_mean(adj, hop1)
    structural = np.column_stack(
        [
            np.log1p(indeg),
            np.log1p(outdeg),
            np.log1p(indeg + outdeg),
            np.log(pr * n),
            np.log1p(nbr_deg),
        ]
    ).astype(np.float32)
    f = x_local.shape[1]
    names = ["log_in_degree", "log_out_degree", "log_degree", "log_pagerank", "log_nbr_degree"]
    names += [f"hop1_mean_{i}" for i in range(f)] + [f"hop2_mean_{i}" for i in range(f)]
    return np.hstack([structural, hop1, hop2]), names
