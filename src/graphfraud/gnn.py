"""GraphSAGE and GAT in plain PyTorch (no torch_geometric needed), trained full-batch.

The graph is treated as undirected. GraphSAGE uses the mean aggregator; GAT uses multi-head
additive attention with a per-destination softmax implemented with ``scatter_reduce``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn

from .boosting import TwoStageResult


def undirected_edges(edge_index: np.ndarray, n: int, self_loops: bool) -> torch.Tensor:
    """Deduplicated undirected (2, E) edge tensor, optionally with self-loops."""
    src, dst = edge_index
    both = np.concatenate([np.vstack([src, dst]), np.vstack([dst, src])], axis=1)
    both = both[:, both[0] != both[1]]
    if self_loops:
        loops = np.arange(n, dtype=np.int64)
        both = np.concatenate([both, np.vstack([loops, loops])], axis=1)
    both = np.unique(both, axis=1)
    return torch.as_tensor(both, dtype=torch.long)


def mean_adjacency(edges: torch.Tensor, n: int) -> torch.Tensor:
    """Sparse row-normalised adjacency: row i averages the neighbours of i."""
    src, dst = edges
    deg = torch.bincount(dst, minlength=n).clamp(min=1).to(torch.float32)
    vals = 1.0 / deg[dst]
    coo = torch.sparse_coo_tensor(
        torch.stack([dst, src]), vals, (n, n), check_invariants=False
    ).coalesce()
    return coo.to_sparse_csr()  # CSR sparse-dense matmul is markedly faster on CPU


class SAGELayer(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.lin_self = nn.Linear(d_in, d_out)
        self.lin_neigh = nn.Linear(d_in, d_out, bias=False)

    def forward(
        self, x: torch.Tensor, adj: torch.Tensor, ax: torch.Tensor | None = None
    ) -> torch.Tensor:
        agg = torch.sparse.mm(adj, x) if ax is None else ax
        return self.lin_self(x) + self.lin_neigh(agg)


class GraphSAGE(nn.Module):
    def __init__(self, d_in: int, hidden: int = 64, layers: int = 2, dropout: float = 0.3):
        super().__init__()
        dims = [d_in] + [hidden] * layers
        pairs = zip(dims[:-1], dims[1:], strict=True)
        self.convs = nn.ModuleList(SAGELayer(a, b) for a, b in pairs)
        self.head = nn.Linear(hidden, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, graph: dict) -> torch.Tensor:
        for i, conv in enumerate(self.convs):
            # the first layer's neighbourhood mean of the raw inputs is fixed: reuse it
            ax = graph.get("ax") if i == 0 else None
            x = self.dropout(torch.relu(conv(x, graph["adj"], ax)))
        return self.head(x).squeeze(-1)


class GATLayer(nn.Module):
    def __init__(self, d_in: int, d_out: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.heads, self.d_out = heads, d_out
        self.lin = nn.Linear(d_in, heads * d_out, bias=False)
        self.att_src = nn.Parameter(torch.empty(heads, d_out))
        self.att_dst = nn.Parameter(torch.empty(heads, d_out))
        self.bias = nn.Parameter(torch.zeros(heads * d_out))
        self.att_dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def attention(self, h: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        """Per-edge attention weights (E, H); they sum to 1 over each destination's in-edges."""
        src, dst = edges
        n = h.shape[0]
        a_src = (h * self.att_src).sum(-1)
        a_dst = (h * self.att_dst).sum(-1)
        e = nn.functional.leaky_relu(a_src[src] + a_dst[dst], 0.2)
        idx = dst.unsqueeze(-1).expand_as(e)
        e_max = torch.full((n, self.heads), -torch.inf, dtype=e.dtype)
        e_max = e_max.scatter_reduce(0, idx, e, reduce="amax", include_self=True)
        w = torch.exp(e - e_max[dst])
        denom = torch.zeros((n, self.heads), dtype=e.dtype).index_add_(0, dst, w)
        return w / denom[dst].clamp(min=1e-16)

    def forward(self, x: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        h = self.lin(x).view(n, self.heads, self.d_out)
        alpha = self.att_dropout(self.attention(h, edges))
        src, dst = edges
        out = torch.zeros_like(h).index_add_(0, dst, h[src] * alpha.unsqueeze(-1))
        return out.reshape(n, self.heads * self.d_out) + self.bias


class GAT(nn.Module):
    def __init__(
        self, d_in: int, hidden: int = 16, heads: int = 4, layers: int = 2, dropout: float = 0.3
    ):
        super().__init__()
        dims = [d_in] + [hidden * heads] * layers
        self.convs = nn.ModuleList(
            GATLayer(a, hidden, heads=heads) for a in dims[:-1]
        )
        self.head = nn.Linear(hidden * heads, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, graph: dict) -> torch.Tensor:
        for conv in self.convs:
            x = self.dropout(nn.functional.elu(conv(x, graph["edges"])))
        return self.head(x).squeeze(-1)


def build_graph_inputs(edge_index: np.ndarray, n: int, kind: str) -> dict:
    if kind == "sage":
        return {"adj": mean_adjacency(undirected_edges(edge_index, n, self_loops=False), n)}
    if kind == "gat":
        return {"edges": undirected_edges(edge_index, n, self_loops=True)}
    raise ValueError(f"unknown GNN kind {kind!r}")


def make_model(kind: str, d_in: int) -> nn.Module:
    return GraphSAGE(d_in) if kind == "sage" else GAT(d_in)


@dataclass
class GNNConfig:
    max_epochs: int = 200
    patience: int = 30
    lr: float = 0.01
    weight_decay: float = 5e-4


def _fit(
    kind: str,
    x: torch.Tensor,
    graph: dict,
    y: torch.Tensor,
    train_mask: np.ndarray,
    cfg: GNNConfig,
    seed: int,
    epochs: int,
    val_mask: np.ndarray | None = None,
) -> tuple[nn.Module, int]:
    torch.manual_seed(seed)
    model = make_model(kind, x.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    tr = torch.as_tensor(np.flatnonzero(train_mask))
    n_pos = float(y[tr].sum())
    pos_weight = torch.tensor(((len(tr) - n_pos) / max(n_pos, 1.0)) ** 0.5)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best_ap, best_epoch, best_state, since = -1.0, epochs, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(x, graph)[tr], y[tr])
        loss.backward()
        opt.step()
        if val_mask is None:
            continue
        p = predict(model, x, graph)[val_mask]
        ap = average_precision_score(y.numpy()[val_mask], p)
        if ap > best_ap:
            best_ap, best_epoch, since = ap, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch


@torch.no_grad()
def predict(model: nn.Module, x: torch.Tensor, graph: dict) -> np.ndarray:
    model.eval()
    return torch.sigmoid(model(x, graph)).numpy()


def train_gnn(
    kind: str,
    x: np.ndarray,
    edge_index: np.ndarray,
    y: np.ndarray,
    masks: dict[str, np.ndarray],
    seed: int = 0,
    cfg: GNNConfig | None = None,
    refit_masks: Sequence[np.ndarray] = (),
) -> TwoStageResult:
    """Select the epoch count on the validation block, then refit on the full training window.

    Unlabelled nodes take part in message passing but never in the loss. ``refit_masks`` are
    extra training windows on which the final recipe (same epochs and seed) is refit.
    """
    cfg = cfg or GNNConfig()
    graph = build_graph_inputs(edge_index, x.shape[0], kind)
    xt = torch.as_tensor(x, dtype=torch.float32)
    if kind == "sage":
        graph["ax"] = torch.sparse.mm(graph["adj"], xt)
    yt = torch.as_tensor(np.clip(y, 0, 1), dtype=torch.float32)
    stage1, best_epoch = _fit(
        kind, xt, graph, yt, masks["fit"], cfg, seed, cfg.max_epochs, masks["val"]
    )
    p_val_stage = predict(stage1, xt, graph)

    def refit(mask: np.ndarray) -> np.ndarray:
        model, _ = _fit(kind, xt, graph, yt, mask, cfg, seed, best_epoch)
        return predict(model, xt, graph)

    return TwoStageResult(
        p_val_stage, refit(masks["train"]), best_epoch, [refit(m) for m in refit_masks]
    )
