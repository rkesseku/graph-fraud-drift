"""End-to-end experiment: features -> five models x seeds -> metrics.json + figures."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .boosting import train_xgb
from .data import load_elliptic
from .gnn import GNNConfig, train_gnn
from .graph_features import graph_features
from .metrics import (
    block_bootstrap_f1,
    cost_optimal_threshold,
    counts_by_step,
    expected_cost,
    illicit_metrics,
    per_step_f1,
)
from .split import TemporalSplit

MODELS = {
    "xgb_local": ("XGBoost, local features (93)", "local"),
    "xgb_all": ("XGBoost, local + aggregated (165)", "all"),
    "xgb_all_graph": ("XGBoost, 165 + engineered graph features", "all_graph"),
    "sage": ("GraphSAGE (165 features)", "all"),
    "gat": ("GAT (165 features)", "all"),
}
REFERENCE = "xgb_all"


@dataclass
class RunConfig:
    seeds: tuple[int, ...] = (0, 1, 2)
    c_fn: float = 10.0  # cost of a missed illicit transaction
    c_fp: float = 1.0  # cost of reviewing a false alert
    split: TemporalSplit = field(default_factory=TemporalSplit)
    gnn: GNNConfig = field(default_factory=GNNConfig)
    xgb_rounds: int = 1000
    n_boot: int = 2000
    models: tuple[str, ...] = tuple(MODELS)


def _standardise(x: np.ndarray, ref_mask: np.ndarray) -> np.ndarray:
    mu = x[ref_mask].mean(axis=0)
    sd = x[ref_mask].std(axis=0)
    sd[sd < 1e-8] = 1.0
    return ((x - mu) / sd).astype(np.float32)


def _mean_sd(values: list[float]) -> dict:
    a = np.asarray(values, dtype=float)
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0}


def run_all(raw_dir: str | Path, out_dir: str | Path, cfg: RunConfig | None = None) -> dict:
    cfg = cfg or RunConfig()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    g = load_elliptic(raw_dir)
    masks = cfg.split.masks(g.step, g.y)
    y01 = np.clip(g.y, 0, 1).astype(np.int64)

    t0 = time.perf_counter()
    gf, _ = graph_features(g.edge_index, g.x_local)
    feat_seconds = time.perf_counter() - t0
    feats = {"local": g.x_local, "all": g.x, "all_graph": np.hstack([g.x, gf])}
    gnn_x = _standardise(g.x, masks["fit"])

    test, val = masks["test"], masks["val"]
    yt, st = y01[test], g.step[test]
    per_model: dict[str, dict] = {}
    step_counts: dict[str, list[np.ndarray]] = {}
    curves: dict[str, np.ndarray] = {}

    for name in cfg.models:
        label, feat_key = MODELS[name]
        runs = []
        for seed in cfg.seeds:
            t0 = time.perf_counter()
            if name.startswith("xgb"):
                p1, pf, size = train_xgb(
                    feats[feat_key], y01, masks, seed=seed, max_rounds=cfg.xgb_rounds
                )
            else:
                p1, pf, size = train_gnn(name, gnn_x, g.edge_index, g.y, masks, seed, cfg.gnn)
            secs = time.perf_counter() - t0
            thr = cost_optimal_threshold(y01[val], p1[val], cfg.c_fn, cfg.c_fp)
            pt = pf[test]
            runs.append(
                {
                    "seed": seed,
                    "size": size,
                    "seconds": secs,
                    "val_auprc_stage1": illicit_metrics(y01[val], p1[val])["auprc"],
                    "test_at_0.5": illicit_metrics(yt, pt, 0.5),
                    "cost_threshold": thr,
                    "test_at_cost_threshold": illicit_metrics(yt, pt, thr),
                    "test_cost_at_0.5": expected_cost(yt, pt, 0.5, cfg.c_fn, cfg.c_fp),
                    "test_cost_at_cost_threshold": expected_cost(
                        yt, pt, thr, cfg.c_fn, cfg.c_fp
                    ),
                    "f1_pre_shutdown": illicit_metrics(
                        y01[masks["test_pre_shutdown"]], pf[masks["test_pre_shutdown"]]
                    )["f1"],
                    "f1_post_shutdown": illicit_metrics(
                        y01[masks["test_post_shutdown"]], pf[masks["test_post_shutdown"]]
                    )["f1"],
                    "per_step_f1": per_step_f1(yt, pt, st, 0.5),
                }
            )
            step_counts.setdefault(name, []).append(counts_by_step(yt, pt >= 0.5, st)[1])
            if seed == cfg.seeds[0]:
                curves[name] = pt
            print(f"[{name}] seed={seed} f1@0.5={runs[-1]['test_at_0.5']['f1']:.3f} "
                  f"({secs:.1f}s)", flush=True)

        steps = sorted(runs[0]["per_step_f1"])
        per_model[name] = {
            "label": label,
            "features": feat_key,
            "summary": {
                "f1": _mean_sd([r["test_at_0.5"]["f1"] for r in runs]),
                "precision": _mean_sd([r["test_at_0.5"]["precision"] for r in runs]),
                "recall": _mean_sd([r["test_at_0.5"]["recall"] for r in runs]),
                "auprc": _mean_sd([r["test_at_0.5"]["auprc"] for r in runs]),
                "auroc": _mean_sd([r["test_at_0.5"]["auroc"] for r in runs]),
                "f1_pre_shutdown": _mean_sd([r["f1_pre_shutdown"] for r in runs]),
                "f1_post_shutdown": _mean_sd([r["f1_post_shutdown"] for r in runs]),
                "cost_at_0.5": _mean_sd([r["test_cost_at_0.5"] for r in runs]),
                "cost_at_cost_threshold": _mean_sd(
                    [r["test_cost_at_cost_threshold"] for r in runs]
                ),
                "recall_at_cost_threshold": _mean_sd(
                    [r["test_at_cost_threshold"]["recall"] for r in runs]
                ),
                "precision_at_cost_threshold": _mean_sd(
                    [r["test_at_cost_threshold"]["precision"] for r in runs]
                ),
                "seconds": _mean_sd([r["seconds"] for r in runs]),
                "per_step_f1_mean": {
                    s: float(np.mean([r["per_step_f1"][s] for r in runs])) for s in steps
                },
            },
            "runs": runs,
        }

    boot = {}
    if REFERENCE in step_counts:
        boot = block_bootstrap_f1(step_counts, REFERENCE, n_boot=cfg.n_boot)
        for name, b in boot.items():
            per_model[name]["bootstrap"] = b

    lab_steps = np.unique(g.step)
    label_counts = {
        int(s): {
            "illicit": int(((g.step == s) & (g.y == 1)).sum()),
            "licit": int(((g.step == s) & (g.y == 0)).sum()),
        }
        for s in lab_steps
    }
    cfg_dict = asdict(cfg)
    result = {
        "dataset": g.summary(),
        "protocol": {
            "fit_steps": f"1-{cfg.split.val_start - 1}",
            "val_steps": f"{cfg.split.val_start}-{cfg.split.train_end}",
            "final_train_steps": f"1-{cfg.split.train_end}",
            "test_steps": f"{cfg.split.train_end + 1}-{int(g.step.max())}",
            "dark_market_step": cfg.split.dark_market_step,
            "n_fit": int(masks["fit"].sum()),
            "n_val": int(val.sum()),
            "n_train": int(masks["train"].sum()),
            "n_test": int(test.sum()),
            "n_test_illicit": int(yt.sum()),
            "costs": {"missed_illicit": cfg.c_fn, "false_alert": cfg.c_fp},
            "bootstrap": f"{cfg.n_boot} resamples of test time steps; reference {REFERENCE}",
            "graph_feature_seconds": feat_seconds,
        },
        "config": {k: v for k, v in cfg_dict.items() if k != "split"},
        "label_counts_by_step": label_counts,
        "models": per_model,
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    from .plots import make_result_figures

    make_result_figures(result, curves, yt, out)
    return result
