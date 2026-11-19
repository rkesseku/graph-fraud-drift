import json

import numpy as np
import pytest
import torch

from graphfraud.cli import main
from graphfraud.data import load_elliptic
from graphfraud.gnn import GATLayer, GraphSAGE, build_graph_inputs, undirected_edges
from graphfraud.graph_features import degrees, graph_features, pagerank, symmetric_adjacency
from graphfraud.metrics import (
    block_bootstrap_f1,
    cost_optimal_threshold,
    counts_by_step,
    expected_cost,
    illicit_metrics,
    prf,
)
from graphfraud.split import TemporalSplit
from graphfraud.testing import write_synthetic_elliptic


@pytest.fixture(scope="module")
def raw(tmp_path_factory):
    return write_synthetic_elliptic(tmp_path_factory.mktemp("raw"), shift_step=11, seed=1)


def test_loader_shapes_and_labels(raw):
    g = load_elliptic(raw, n_local=6)
    s = g.summary()
    assert s["nodes"] == 12 * 120
    assert g.x.shape == (1440, 10) and g.x_local.shape == (1440, 6)
    assert set(np.unique(g.y)) <= {-1, 0, 1}
    assert g.edge_index.max() < g.num_nodes and g.num_edges > 0
    assert s["illicit"] + s["licit"] + s["unknown"] == s["nodes"]


def test_loader_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_elliptic(tmp_path)


def test_temporal_masks_are_disjoint_and_ordered():
    step = np.repeat(np.arange(1, 13), 5)
    y = np.tile([1, 0, -1, 0, 1], 12)
    m = TemporalSplit(val_start=7, train_end=8, dark_market_step=11).masks(step, y)
    assert not (m["fit"] & m["val"]).any() and not (m["train"] & m["test"]).any()
    assert (m["fit"] | m["val"]).sum() == m["train"].sum()
    assert step[m["train"]].max() < step[m["test"]].min()
    assert (y[m["test"]] >= 0).all()
    assert (m["test_pre_shutdown"] | m["test_post_shutdown"]).sum() == m["test"].sum()
    with pytest.raises(ValueError):
        TemporalSplit(val_start=9, train_end=8).masks(step, y)


def test_degrees_pagerank_and_adjacency():
    ei = np.array([[0, 1, 2, 2], [1, 2, 0, 3]])
    indeg, outdeg = degrees(ei, 4)
    assert indeg.tolist() == [1, 1, 1, 1] and outdeg.tolist() == [1, 1, 2, 0]
    pr = pagerank(ei, 4)
    assert pr.sum() == pytest.approx(1.0) and (pr > 0).all()
    adj = symmetric_adjacency(np.array([[0, 0, 1], [1, 1, 1]]), 3)  # duplicate + self-loop
    assert adj.nnz == 2 and adj[0, 1] == 1 and adj[1, 1] == 0


def test_graph_features_shape_and_finite(raw):
    g = load_elliptic(raw, n_local=6)
    f, names = graph_features(g.edge_index, g.x_local)
    assert f.shape == (g.num_nodes, 5 + 12) and len(names) == f.shape[1]
    assert np.isfinite(f).all()


def test_gat_attention_sums_to_one():
    ei = np.array([[0, 1, 2, 3], [1, 2, 3, 0]])
    edges = undirected_edges(ei, 5, self_loops=True)  # node 4 is isolated
    layer = GATLayer(3, 2, heads=2)
    h = layer.lin(torch.randn(5, 3)).view(5, 2, 2)
    alpha = layer.attention(h, edges)
    sums = torch.zeros(5, 2).index_add_(0, edges[1], alpha)
    assert torch.allclose(sums, torch.ones(5, 2), atol=1e-6)


def test_sage_forward_shape():
    ei = np.array([[0, 1, 2], [1, 2, 0]])
    graph = build_graph_inputs(ei, 4, "sage")
    out = GraphSAGE(3, hidden=8)(torch.randn(4, 3), graph)
    assert out.shape == (4,)


def test_metrics_and_cost_threshold():
    y = np.array([1, 1, 0, 0, 0, 0])
    p = np.array([0.9, 0.3, 0.4, 0.1, 0.2, 0.05])
    m = illicit_metrics(y, p, 0.5)
    assert (m["tp"], m["fp"], m["fn"]) == (1, 0, 1)
    assert m["f1"] == pytest.approx(2 / 3)
    assert prf(0, 0, 0) == (0.0, 0.0, 0.0)
    # expensive misses push the threshold down to catch the second illicit case
    thr = cost_optimal_threshold(y, p, c_fn=10, c_fp=1)
    assert thr <= 0.3
    assert expected_cost(y, p, thr, 10, 1) < expected_cost(y, p, 0.5, 10, 1)


def test_block_bootstrap_identical_models_zero_diff():
    y = np.array([1, 0, 1, 0, 1, 0, 0, 1])
    step = np.array([1, 1, 2, 2, 3, 3, 4, 4])
    pred = np.array([1, 0, 0, 0, 1, 1, 0, 1])
    _, c = counts_by_step(y, pred, step)
    out = block_bootstrap_f1({"a": [c], "b": [c.copy()]}, "a", n_boot=200)
    assert out["b"]["diff_vs_reference"] == 0.0
    lo, hi = out["a"]["f1_ci95"]
    assert 0.0 <= lo <= hi <= 1.0


def test_run_all_end_to_end(raw, tmp_path):
    out = tmp_path / "results"
    main([
        "run-all", "--raw-dir", str(raw), "--out-dir", str(out), "--seeds", "0",
        "--val-start", "7", "--train-end", "8", "--dark-market-step", "11",
        "--max-epochs", "30", "--n-boot", "100",
    ])
    res = json.loads((out / "metrics.json").read_text())
    assert set(res["models"]) == {"xgb_local", "xgb_all", "xgb_all_graph", "sage", "gat"}
    for m in res["models"].values():
        assert 0.0 <= m["summary"]["f1"]["mean"] <= 1.0
        assert "bootstrap" in m
    # the synthetic signal is easy before the drift step
    assert res["models"]["xgb_all"]["summary"]["f1_pre_shutdown"]["mean"] > 0.6
    for fig in ("per_step_f1.png", "pr_curves.png", "f1_comparison.png"):
        assert (out / fig).exists()
