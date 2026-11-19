import json

import numpy as np
import pytest
import torch
from xgboost import XGBClassifier

from graphfraud.boosting import DEFAULT_PARAMS, TwoStageResult, train_xgb
from graphfraud.cli import main
from graphfraud.data import load_elliptic
from graphfraud.gnn import GATLayer, GraphSAGE, build_graph_inputs, undirected_edges
from graphfraud.graph_features import degrees, graph_features, pagerank, symmetric_adjacency
from graphfraud.metrics import (
    block_bootstrap_f1,
    block_bootstrap_paired,
    cost_optimal_threshold,
    counts_by_step,
    expected_cost,
    illicit_metrics,
    prf,
)
from graphfraud.pipeline import rolling_origin_windows, select_thresholds
from graphfraud.split import TemporalSplit
from graphfraud.testing import write_synthetic_elliptic
from graphfraud.tuning import DEFAULT_TRIAL, tune_xgb


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


def test_rolling_threshold_lives_on_the_deployed_models_scale():
    # Deployed (refit) scores are a monotone transform of the stage-1 scale. The rolling-origin
    # threshold must then give exactly the decisions that the stage-1 threshold gives on its own
    # model, while the old mismatched threshold (stage-1 value applied to refit scores) does not.
    rng = np.random.default_rng(0)
    step = np.repeat(np.arange(1, 11), 200)
    y = (rng.random(step.size) < 0.15).astype(np.int64)
    q = np.clip(0.25 + 0.4 * y + rng.normal(0, 0.15, step.size), 1e-3, 1 - 1e-3)
    final = q**4  # same ranking, different scale
    split = TemporalSplit(val_start=6, train_end=8, dark_market_step=10)
    masks = split.masks(step, y)
    windows = rolling_origin_windows(step, y, split)
    assert [o for o, _ in windows] == [6, 7, 8]
    assert all(step[m].max() == o - 1 for o, m in windows)
    res = TwoStageResult(q, final, 1, [final] * len(windows))
    thr = select_thresholds(y, step, masks, res, [o for o, _ in windows], 10.0, 1.0)
    assert thr["mismatched"] == cost_optimal_threshold(y[masks["val"]], q[masks["val"]], 10, 1)
    test = masks["test"]
    own = q[test] >= thr["mismatched"]
    assert ((final[test] >= thr["rolling_origin"]) == own).all()
    assert ((final[test] >= thr["mismatched"]) != own).any()


def test_threshold_chosen_from_same_recipe_as_evaluated_model(raw):
    g = load_elliptic(raw, n_local=6)
    y01 = np.clip(g.y, 0, 1)
    split = TemporalSplit(val_start=7, train_end=8, dark_market_step=11)
    masks = split.masks(g.step, g.y)
    windows = rolling_origin_windows(g.step, g.y, split)
    res = train_xgb(g.x, y01, masks, seed=0, max_rounds=60, refit_masks=[m for _, m in windows],
                    n_jobs=1)
    assert len(res.p_refits) == len(windows)

    def recipe(mask):
        m = XGBClassifier(n_estimators=res.size, random_state=0, n_jobs=1, **DEFAULT_PARAMS)
        return m.fit(g.x[mask], y01[mask], verbose=False).predict_proba(g.x)[:, 1]

    # the evaluated model and every threshold-selection model come from the same recipe
    np.testing.assert_allclose(res.p_final, recipe(masks["train"]), rtol=1e-6)
    oot = np.empty(len(y01))
    for (o, m), p in zip(windows, res.p_refits, strict=True):
        np.testing.assert_allclose(p, recipe(m), rtol=1e-6)
        oot[g.step == o] = p[g.step == o]
    val = masks["val"]
    thr = select_thresholds(y01, g.step, masks, res, [o for o, _ in windows], 10.0, 1.0)
    assert thr["rolling_origin"] == cost_optimal_threshold(y01[val], oot[val], 10, 1)
    assert thr["mismatched"] == cost_optimal_threshold(y01[val], res.p_stage1[val], 10, 1)


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
        tv = m["summary"]["threshold_variants"]
        assert {"mismatched", "rolling_origin", "stage1_own"} <= set(tv)
        assert all(0.0 <= tv[k]["cost"]["mean"] for k in ("mismatched", "rolling_origin"))
    assert res["protocol"]["threshold_origins"] == [7, 8]
    # the synthetic signal is easy before the drift step
    assert res["models"]["xgb_all"]["summary"]["f1_pre_shutdown"]["mean"] > 0.6
    for fig in ("per_step_f1.png", "pr_curves.png", "f1_comparison.png"):
        assert (out / fig).exists()


def test_paired_bootstrap_identical_arms_and_cost_direction():
    rng = np.random.default_rng(0)
    step = np.repeat(np.arange(1, 7), 50)
    y = (rng.random(step.size) < 0.2).astype(int)
    p = np.clip(0.3 * y + rng.random(step.size) * 0.7, 0, 1)
    arms = {"default": [(p, 0.5)], "same": [(p.copy(), 0.5)], "never": [(np.zeros_like(p), 0.5)]}

    def cost(yy, pp, t):
        return expected_cost(yy, pp, t, 10.0, 1.0)

    out = block_bootstrap_paired(y, step, arms, "default", cost, n_boot=200)
    assert out["same"]["diff_vs_reference"] == 0.0 and out["same"]["diff_ci95"] == [0.0, 0.0]
    assert out["never"]["estimate"] == pytest.approx(10 * y.mean())
    lo, hi = out["default"]["ci95"]
    assert lo <= out["default"]["estimate"] <= hi


def test_tuning_objective_never_reads_test_rows(raw):
    pytest.importorskip("optuna")
    g = load_elliptic(raw, n_local=6)
    y01 = np.clip(g.y, 0, 1).astype(np.int64)
    masks = TemporalSplit(val_start=7, train_end=8, dark_market_step=11).masks(g.step, g.y)
    kw = {"n_trials": 4, "seed": 0, "max_rounds": 40, "early_stopping": 10, "n_jobs": 1,
          "verbose": False}
    clean = tune_xgb(g.x, y01, masks, **kw)
    # Corrupt everything outside the fit and validation blocks (test steps and unlabelled rows):
    # the search must be unaffected.
    x_bad, y_bad = g.x.copy(), y01.copy()
    outside = ~(masks["fit"] | masks["val"])
    x_bad[outside] = np.nan
    y_bad[outside] = 1 - y_bad[outside]
    corrupted = tune_xgb(x_bad, y_bad, masks, **kw)
    vals = [t["val_auprc"] for t in clean["trials"]]
    assert vals == [t["val_auprc"] for t in corrupted["trials"]]
    assert clean["best_params"] == corrupted["best_params"]
    # trial 0 is the fixed default recipe, so tuned >= default on validation
    assert clean["trials"][0]["params"] == DEFAULT_TRIAL
    assert clean["best_val_auprc"] >= clean["default_val_auprc"]
    bad = dict(masks, val=masks["val"] | masks["test"])
    with pytest.raises(ValueError):
        tune_xgb(g.x, y01, bad, **kw)


def test_tune_cli_end_to_end(raw, tmp_path):
    pytest.importorskip("optuna")
    out = tmp_path / "tune"
    main([
        "tune", "--raw-dir", str(raw), "--out-dir", str(out), "--seeds", "0", "1",
        "--val-start", "7", "--train-end", "8", "--dark-market-step", "11",
        "--n-trials", "3", "--xgb-rounds", "40", "--n-boot", "50", "--n-jobs", "1",
        "--extra-n-jobs", "2",
    ])
    res = json.loads((out / "tuning.json").read_text())
    m = res["models"]["xgb_all"]
    assert set(m["arms"]) == {"default", "tuned", "default_2t", "tuned_2t"}
    assert m["arms"]["tuned_2t"]["n_jobs"] == 2
    assert m["search"]["n_trials"] == 3 and res["protocol"]["n_jobs"] == 1
    assert set(m["comparisons"]) == {"default", "default_2t"}
    c = m["comparisons"]["default"]
    assert {"f1_at_0.5", "auprc", "cost_at_0.5", "cost_rolling_origin"} <= set(c)
    assert c["auprc"]["default"]["diff_vs_reference"] == 0.0
    assert "Does tuning help?" in (out / "tuning.md").read_text()
