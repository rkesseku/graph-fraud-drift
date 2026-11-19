"""Optuna search for XGBoost hyperparameters on the validation block, then a tuned-vs-default test.

The search only ever sees the fit block (steps before ``val_start``) and the validation block
(``val_start``..``train_end``): each trial trains on the fit block, early-stops on validation
AUPRC and is scored by validation AUPRC. Trial 0 is the repo's fixed default recipe, so the best
trial is at least as good as the default *on validation*. The tuned recipe is then run through
the standard protocol (3 seeds, refit on the full training window with the early-stopped tree
count, rolling-origin and original cost thresholds) next to the default recipe, and the test
steps are scored once for each.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

from .boosting import DEFAULT_PARAMS, TwoStageResult, default_n_jobs, train_xgb
from .data import load_elliptic
from .graph_features import graph_features
from .metrics import block_bootstrap_f1, block_bootstrap_paired, expected_cost
from .pipeline import (
    MODELS,
    RunConfig,
    evaluate_seeds,
    rolling_origin_windows,
    summarise_runs,
)

# The fixed recipe written out in full (XGBoost's own defaults for the last three), enqueued as
# trial 0 so the search starts from it.
DEFAULT_TRIAL = {
    "max_depth": DEFAULT_PARAMS["max_depth"],
    "learning_rate": DEFAULT_PARAMS["learning_rate"],
    "min_child_weight": DEFAULT_PARAMS["min_child_weight"],
    "subsample": DEFAULT_PARAMS["subsample"],
    "colsample_bytree": DEFAULT_PARAMS["colsample_bytree"],
    "reg_lambda": 1.0,
    "gamma": 0.0,
    "scale_pos_weight": 1.0,
}


def suggest_params(trial) -> dict:
    """Search space; every range contains the default recipe."""
    return {
        "max_depth": trial.suggest_int("max_depth", 2, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "min_child_weight": trial.suggest_float("min_child_weight", 0.5, 20.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 100.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 10.0, log=True),
    }


def tune_xgb(
    x: np.ndarray,
    y: np.ndarray,
    masks: dict[str, np.ndarray],
    n_trials: int = 60,
    seed: int = 0,
    model_seed: int = 0,
    max_rounds: int = 1000,
    early_stopping: int = 50,
    n_jobs: int | None = None,
    timeout: float | None = None,
    verbose: bool = True,
) -> dict:
    """TPE search maximising validation AUPRC. Only ``masks['fit']`` and ``masks['val']`` rows are
    read; test rows are never touched.

    Returns the best parameters, the validation AUPRC of the best and of the default trial, and
    one record per trial.
    """
    import optuna

    fit, val = masks["fit"], masks["val"]
    if "test" in masks and ((fit | val) & masks["test"]).any():
        raise ValueError("fit/validation rows overlap the test rows")
    x_fit, y_fit, x_val, y_val = x[fit], y[fit], x[val], y[val]
    n_jobs = n_jobs or default_n_jobs()

    def objective(trial) -> float:
        params = suggest_params(trial)
        t0 = time.perf_counter()
        model = XGBClassifier(
            n_estimators=max_rounds,
            early_stopping_rounds=early_stopping,
            random_state=model_seed,
            n_jobs=n_jobs,
            **{**DEFAULT_PARAMS, **params},
        )
        model.fit(x_fit, y_fit, eval_set=[(x_val, y_val)], verbose=False)
        value = float(average_precision_score(y_val, model.predict_proba(x_val)[:, 1]))
        trial.set_user_attr("rounds", int(model.best_iteration) + 1)
        trial.set_user_attr("seconds", time.perf_counter() - t0)
        return value

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed)
    )
    study.enqueue_trial(DEFAULT_TRIAL)

    def report(study, trial) -> None:
        if verbose and (trial.number % 10 == 0 or trial.number == n_trials - 1):
            print(f"  trial {trial.number:3d} val AUPRC {trial.value:.4f} "
                  f"(best {study.best_value:.4f}, {trial.user_attrs['rounds']} trees, "
                  f"{trial.user_attrs['seconds']:.1f}s)", flush=True)

    t0 = time.perf_counter()
    study.optimize(objective, n_trials=n_trials, timeout=timeout, callbacks=[report])
    seconds = time.perf_counter() - t0
    trials = [
        {
            "number": t.number,
            "val_auprc": t.value,
            "params": t.params,
            "rounds": t.user_attrs.get("rounds"),
            "seconds": t.user_attrs.get("seconds"),
        }
        for t in study.trials
        if t.value is not None
    ]
    best = study.best_trial
    return {
        "best_params": dict(best.params),
        "best_trial": best.number,
        "best_val_auprc": float(best.value),
        "best_rounds": best.user_attrs["rounds"],
        "default_val_auprc": float(trials[0]["val_auprc"]),
        "default_rounds": trials[0]["rounds"],
        "n_trials": len(trials),
        "seconds": seconds,
        "trials": trials,
    }


@dataclass
class TuneConfig:
    run: RunConfig = field(default_factory=RunConfig)
    models: tuple[str, ...] = ("xgb_all",)
    n_trials: int = 60
    sampler_seed: int = 0
    model_seed: int = 0  # XGBoost seed used inside the search
    timeout: float | None = None
    # extra thread counts at which both recipes are also evaluated (XGBoost results depend on it)
    extra_n_jobs: tuple[int, ...] = ()


def _cost(c_fn: float, c_fp: float):
    def metric(y: np.ndarray, p: np.ndarray, thr: float) -> float:
        return expected_cost(y, p, thr, c_fn, c_fp)

    return metric


def _auprc(y: np.ndarray, p: np.ndarray, _thr: float) -> float:
    return float(average_precision_score(y, p))


def compare_arms(
    y01: np.ndarray,
    step: np.ndarray,
    test: np.ndarray,
    arms: dict[str, dict],
    cfg: RunConfig,
    reference: str = "default",
) -> dict:
    """Paired time-step bootstrap of every arm against ``reference`` on the test steps."""
    yt, st = y01[test], step[test]
    counts = {name: a["counts"] for name, a in arms.items()}
    f1 = block_bootstrap_f1(counts, reference, n_boot=cfg.n_boot)
    out = {"f1_at_0.5": f1}

    def pairs(name: str, thr_key: str | None) -> list[tuple[np.ndarray, float]]:
        return [
            (p, 0.5 if thr_key is None else r["thresholds"][thr_key])
            for p, r in zip(arms[name]["test_scores"], arms[name]["runs"], strict=True)
        ]

    cost = _cost(cfg.c_fn, cfg.c_fp)
    specs = {"auprc": (_auprc, None), "cost_at_0.5": (cost, None)}
    if all("rolling_origin" in r["thresholds"] for a in arms.values() for r in a["runs"]):
        specs["cost_rolling_origin"] = (cost, "rolling_origin")
    specs["cost_original_threshold"] = (cost, "mismatched")
    for key, (metric, thr_key) in specs.items():
        out[key] = block_bootstrap_paired(
            yt, st, {name: pairs(name, thr_key) for name in arms}, reference, metric,
            n_boot=cfg.n_boot,
        )
    return out


def run_tuning(raw_dir: str | Path, out_dir: str | Path, cfg: TuneConfig | None = None) -> dict:
    """Tune each requested XGBoost model on validation, then compare tuned vs default on test."""
    cfg = cfg or TuneConfig()
    rc = cfg.run
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    g = load_elliptic(raw_dir)
    masks = rc.split.masks(g.step, g.y)
    y01 = np.clip(g.y, 0, 1).astype(np.int64)
    feats = {"local": g.x_local, "all": g.x}
    if any(MODELS[m][1] == "all_graph" for m in cfg.models):
        gf, _ = graph_features(g.edge_index, g.x_local)
        feats["all_graph"] = np.hstack([g.x, gf])
    windows = rolling_origin_windows(g.step, g.y, rc.split) if rc.rolling_threshold else []
    origins = [o for o, _ in windows]
    refit_masks = [m for _, m in windows]
    test = masks["test"]

    models: dict[str, dict] = {}
    t_all = time.perf_counter()
    for name in cfg.models:
        label, feat_key = MODELS[name]
        if not name.startswith("xgb"):
            raise ValueError(f"tuning is implemented for the XGBoost models only, not {name}")
        x = feats[feat_key]
        print(f"[{name}] tuning on validation steps "
              f"{rc.split.val_start}-{rc.split.train_end} ({cfg.n_trials} trials)", flush=True)
        search = tune_xgb(
            x, y01, masks, n_trials=cfg.n_trials, seed=cfg.sampler_seed,
            model_seed=cfg.model_seed, max_rounds=rc.xgb_rounds, n_jobs=rc.n_jobs,
            timeout=cfg.timeout,
        )
        arms: dict[str, dict] = {}
        recipes = [("default", None, rc.n_jobs), ("tuned", search["best_params"], rc.n_jobs)]
        for nj in cfg.extra_n_jobs:
            recipes += [(f"default_{nj}t", None, nj), (f"tuned_{nj}t", search["best_params"], nj)]
        for arm, params, nj in recipes:

            def train_fn(seed: int, x=x, params=params, nj=nj) -> TwoStageResult:
                return train_xgb(
                    x, y01, masks, seed=seed, max_rounds=rc.xgb_rounds, params=params,
                    refit_masks=refit_masks, n_jobs=nj,
                )

            scores: list[np.ndarray] = []

            def keep(seed: int, train_fn=train_fn, scores=scores) -> TwoStageResult:
                res = train_fn(seed)
                scores.append(res.p_final[test])
                return res

            runs, counts, _ = evaluate_seeds(
                f"{name}/{arm}", keep, y01, g.step, masks, origins, rc
            )
            arms[arm] = {"runs": runs, "counts": counts, "test_scores": scores, "n_jobs": nj}
        references = [a for a in arms if a.startswith("default")]
        comparisons = {
            ref: compare_arms(y01, g.step, test, arms, rc, reference=ref) for ref in references
        }
        models[name] = {
            "label": label,
            "features": feat_key,
            "search": search,
            "default_params": {**DEFAULT_PARAMS, **DEFAULT_TRIAL},
            "tuned_params": {**DEFAULT_PARAMS, **search["best_params"]},
            "arms": {
                arm: {"n_jobs": a["n_jobs"], "summary": summarise_runs(a["runs"]),
                      "runs": a["runs"]}
                for arm, a in arms.items()
            },
            "comparisons": comparisons,
        }

    run_cfg = asdict(rc)
    result = {
        "protocol": {
            "search_train_steps": f"1-{rc.split.val_start - 1}",
            "search_val_steps": f"{rc.split.val_start}-{rc.split.train_end}",
            "objective": "validation AUPRC of a model trained on the fit block, early-stopped "
            "on the validation block (test steps never read)",
            "sampler": f"TPE (seed {cfg.sampler_seed}); trial 0 = default recipe; no pruning",
            "final_train_steps": f"1-{rc.split.train_end}",
            "test_steps": f"{rc.split.train_end + 1}-{int(g.step.max())}",
            "seeds": list(rc.seeds),
            "n_jobs": rc.n_jobs,
            "bootstrap": f"{rc.n_boot} paired resamples of test time steps; reference default",
            "costs": {"missed_illicit": rc.c_fn, "false_alert": rc.c_fp},
            "total_seconds": time.perf_counter() - t_all,
        },
        "config": {
            "n_trials": cfg.n_trials,
            "sampler_seed": cfg.sampler_seed,
            "model_seed": cfg.model_seed,
            "timeout": cfg.timeout,
            "xgb_rounds": rc.xgb_rounds,
            "n_jobs": rc.n_jobs,
            "extra_n_jobs": list(cfg.extra_n_jobs),
            "rolling_threshold": rc.rolling_threshold,
            "split": run_cfg["split"],
        },
        "models": models,
    }
    (out / "tuning.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (out / "tuning.md").write_text(tuning_markdown(result), encoding="utf-8")
    return result


def _fmt_ci(d: dict, nd: int = 3) -> str:
    lo, hi = d["diff_ci95"]
    return f"{d['diff_vs_reference']:+.{nd}f} [{lo:+.{nd}f}, {hi:+.{nd}f}]"


def tuning_markdown(result: dict) -> str:
    """Human-readable summary of ``tuning.json``."""
    p = result["protocol"]
    lines = [
        "# Does tuning help? Optuna search for XGBoost",
        "",
        f"Search: {p['sampler']}, objective = {p['objective']}; trains on steps "
        f"{p['search_train_steps']}, validates on {p['search_val_steps']}. Both recipes then run "
        f"the standard protocol (seeds {p['seeds']}, refit on {p['final_train_steps']}, "
        f"thresholds from validation only) and are scored once on test steps {p['test_steps']}. "
        f"Threads: {p['n_jobs']} (arms suffixed `_Nt`: N threads). "
        f"Total time {p['total_seconds'] / 60:.1f} min.",
        "",
    ]
    for name, m in result["models"].items():
        s = m["search"]
        lines += [
            f"## {m['label']} (`{name}`)",
            "",
            f"{s['n_trials']} trials in {s['seconds'] / 60:.1f} min. Validation AUPRC (search "
            f"seed): default {s['default_val_auprc']:.4f} ({s['default_rounds']} trees), best "
            f"{s['best_val_auprc']:.4f} (trial {s['best_trial']}, {s['best_rounds']} trees).",
            "",
            "Tuned parameters: "
            + ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                        for k, v in s["best_params"].items()),
            "",
            "| Recipe | Val AUPRC (stage 1) | Trees | F1 at 0.5 | Precision | Recall | AUPRC "
            "| Cost at 0.5 | Cost, rolling-origin thr. | Cost, original thr. | F1 steps 43+ |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for arm, a in m["arms"].items():
            sm, runs = a["summary"], a["runs"]
            tv = sm["threshold_variants"]
            ro = f"{tv['rolling_origin']['cost']['mean']:.4f}" if "rolling_origin" in tv else "-"
            lines.append(
                f"| {arm} | {np.mean([r['val_auprc_stage1'] for r in runs]):.4f} "
                f"| {', '.join(str(r['size']) for r in runs)} "
                f"| {sm['f1']['mean']:.3f} ± {sm['f1']['sd']:.3f} "
                f"| {sm['precision']['mean']:.3f} | {sm['recall']['mean']:.3f} "
                f"| {sm['auprc']['mean']:.3f} ± {sm['auprc']['sd']:.3f} "
                f"| {sm['cost_at_0.5']['mean']:.4f} | {ro} "
                f"| {tv['mismatched']['cost']['mean']:.4f} "
                f"| {sm['f1_post_shutdown']['mean']:.3f} |"
            )
        labels = {
            "f1_at_0.5": ("F1 at 0.5", 3),
            "auprc": ("AUPRC", 3),
            "cost_at_0.5": ("cost at 0.5", 4),
            "cost_rolling_origin": ("cost, rolling-origin threshold", 4),
            "cost_original_threshold": ("cost, original threshold", 4),
        }
        for ref, c in m["comparisons"].items():
            lines += [
                "",
                f"Differences against `{ref}` ({m['arms'][ref]['n_jobs']} threads), 95% paired "
                "time-step bootstrap interval (lower cost is better):",
                "",
                "| Recipe | " + " | ".join(lab for lab, _ in labels.values()) + " |",
                "|---|" + "---|" * len(labels),
            ]
            for arm in m["arms"]:
                if arm.startswith("tuned"):
                    cells = [_fmt_ci(c[k][arm], nd) if k in c else "-"
                             for k, (_, nd) in labels.items()]
                    lines.append(f"| {arm} - {ref} | " + " | ".join(cells) + " |")
        lines.append("")
    return "\n".join(lines)
