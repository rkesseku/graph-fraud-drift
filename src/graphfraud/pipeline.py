"""End-to-end experiment: features -> five models x seeds -> metrics.json + figures."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .boosting import TwoStageResult, train_xgb
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
    rolling_threshold: bool = True  # refit the final recipe on earlier windows for the threshold
    n_jobs: int | None = 2  # threads for XGBoost / torch; results depend on it (reported run: 2)


def _standardise(x: np.ndarray, ref_mask: np.ndarray) -> np.ndarray:
    mu = x[ref_mask].mean(axis=0)
    sd = x[ref_mask].std(axis=0)
    sd[sd < 1e-8] = 1.0
    return ((x - mu) / sd).astype(np.float32)


def _mean_sd(values: list[float]) -> dict:
    a = np.asarray(values, dtype=float)
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0}


def rolling_origin_windows(
    step: np.ndarray, y: np.ndarray, split: TemporalSplit
) -> list[tuple[int, np.ndarray]]:
    """(origin, training mask) for every validation step: train on labelled steps before it."""
    lab = y >= 0
    return [(o, lab & (step < o)) for o in range(split.val_start, split.train_end + 1)]


def select_thresholds(
    y01: np.ndarray,
    step: np.ndarray,
    masks: dict[str, np.ndarray],
    res: TwoStageResult,
    origins: list[int],
    c_fn: float,
    c_fp: float,
) -> dict[str, float]:
    """Cost-optimal thresholds chosen on the validation steps only (test data never used).

    ``mismatched``: chosen on the stage-1 model's validation scores, the original procedure. It
    is self-consistent for the stage-1 model but is applied to the *refit* model, whose score
    scale can differ.
    ``rolling_origin``: chosen on out-of-time scores of models refit with the final recipe;
    each validation step ``o`` is scored by the refit on labelled steps ``< o`` (``res.p_refits``
    in the order of ``origins``). These are one-step-ahead scores from the same recipe as the
    deployed model, so the threshold lives on the deployed model's score scale.
    """
    val = masks["val"]
    out = {"mismatched": cost_optimal_threshold(y01[val], res.p_stage1[val], c_fn, c_fp)}
    if res.p_refits:
        if len(res.p_refits) != len(origins):
            raise ValueError("need one refit per origin")
        oot = np.full(len(y01), np.nan)
        for o, p in zip(origins, res.p_refits, strict=True):
            sel = val & (step == o)
            oot[sel] = p[sel]
        if np.isnan(oot[val]).any():
            raise ValueError("origins must cover every validation step")
        out["rolling_origin"] = cost_optimal_threshold(y01[val], oot[val], c_fn, c_fp)
    return out


def _cost_block(y: np.ndarray, p: np.ndarray, thr: float, c_fn: float, c_fp: float) -> dict:
    m = illicit_metrics(y, p, thr)
    return {
        "threshold": float(thr),
        "cost": expected_cost(y, p, thr, c_fn, c_fp),
        "recall": m["recall"],
        "precision": m["precision"],
        "f1": m["f1"],
        "alerts": m["tp"] + m["fp"],
    }


def _summarise_variants(runs: list[dict]) -> dict:
    """Mean/sd over seeds of every threshold variant's cost, recall, precision and threshold."""
    keys = ("threshold", "cost", "recall", "precision", "f1")
    first = runs[0]["threshold_variants"]
    out: dict[str, dict] = {}
    for name in first:
        if name == "stage1_own":
            out[name] = {
                part: {k: _mean_sd([r["threshold_variants"][name][part][k] for r in runs])
                       for k in keys}
                for part in ("at_0.5", "at_threshold")
            }
            out[name]["auprc"] = _mean_sd([r["threshold_variants"][name]["auprc"] for r in runs])
        else:
            out[name] = {k: _mean_sd([r["threshold_variants"][name][k] for r in runs])
                         for k in keys}
    return out


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
    windows = rolling_origin_windows(g.step, g.y, cfg.split) if cfg.rolling_threshold else []
    origins = [o for o, _ in windows]
    refit_masks = [m for _, m in windows]
    if cfg.n_jobs:
        import torch

        torch.set_num_threads(cfg.n_jobs)
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
                res = train_xgb(
                    feats[feat_key], y01, masks, seed=seed, max_rounds=cfg.xgb_rounds,
                    refit_masks=refit_masks, n_jobs=cfg.n_jobs,
                )
            else:
                res = train_gnn(
                    name, gnn_x, g.edge_index, g.y, masks, seed, cfg.gnn, refit_masks=refit_masks
                )
            p1, pf, size = res.p_stage1, res.p_final, res.size
            secs = time.perf_counter() - t0
            thr = select_thresholds(y01, g.step, masks, res, origins, cfg.c_fn, cfg.c_fp)
            pt = pf[test]
            variants = {
                # original procedure: stage-1 threshold applied to the refit model
                "mismatched": _cost_block(yt, pt, thr["mismatched"], cfg.c_fn, cfg.c_fp),
                # stage-1 model kept as the deployed model, with its own threshold
                "stage1_own": {
                    "at_0.5": _cost_block(yt, p1[test], 0.5, cfg.c_fn, cfg.c_fp),
                    "at_threshold": _cost_block(
                        yt, p1[test], thr["mismatched"], cfg.c_fn, cfg.c_fp
                    ),
                    "auprc": illicit_metrics(yt, p1[test])["auprc"],
                },
            }
            if "rolling_origin" in thr:
                variants["rolling_origin"] = _cost_block(
                    yt, pt, thr["rolling_origin"], cfg.c_fn, cfg.c_fp
                )
            runs.append(
                {
                    "seed": seed,
                    "size": size,
                    "seconds": secs,
                    "val_auprc_stage1": illicit_metrics(y01[val], p1[val])["auprc"],
                    "test_at_0.5": illicit_metrics(yt, pt, 0.5),
                    "test_cost_at_0.5": expected_cost(yt, pt, 0.5, cfg.c_fn, cfg.c_fp),
                    "thresholds": thr,
                    "threshold_variants": variants,
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
            costs = " ".join(f"{k}={v['cost']:.4f}" for k, v in variants.items() if "cost" in v)
            print(f"[{name}] seed={seed} f1@0.5={runs[-1]['test_at_0.5']['f1']:.3f} "
                  f"cost@0.5={runs[-1]['test_cost_at_0.5']:.4f} {costs} ({secs:.1f}s)",
                  flush=True)

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
                "threshold_variants": _summarise_variants(runs),
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
            "threshold_origins": origins,
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
