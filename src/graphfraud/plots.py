"""Result figures written next to metrics.json."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

COLORS = {
    "xgb_local": "#9aa5b1",
    "xgb_all": "#1f4e9e",
    "xgb_all_graph": "#2a9d8f",
    "sage": "#e76f51",
    "gat": "#c9184a",
}


def _style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.6)


def per_step_figure(result: dict, path: Path) -> None:
    models = result["models"]
    dark = result["protocol"]["dark_market_step"]
    counts = {int(k): v for k, v in result["label_counts_by_step"].items()}  # JSON keys are str
    fig, ax = plt.subplots(figsize=(8, 3.6))
    first = next(iter(models.values()))
    steps = sorted(int(s) for s in first["summary"]["per_step_f1_mean"])
    ax2 = ax.twinx()
    ax2.bar(steps, [counts[s]["illicit"] for s in steps], color="#d9d9d9", width=0.7, zorder=0)
    ax2.set_ylabel("illicit transactions (bars)", color="#777777")
    ax2.spines[["top"]].set_visible(False)
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    for name, m in models.items():
        f1 = {int(k): v for k, v in m["summary"]["per_step_f1_mean"].items()}
        ax.plot(steps, [f1[s] for s in steps], marker="o", ms=3, lw=1.6,
                color=COLORS.get(name), label=m["label"])
    ax.axvline(dark - 0.5, color="black", ls="--", lw=1)
    ax.text(dark - 0.3, 1.02, "dark-market shutdown", fontsize=8, va="bottom")
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("test time step")
    ax.set_ylabel("illicit F1 (threshold 0.5)")
    _style(ax)
    ax.legend(fontsize=7, loc="upper right", bbox_to_anchor=(0.99, 0.9), frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def pr_figure(result: dict, curves: dict, y_test: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    for name, p in curves.items():
        prec, rec, _ = precision_recall_curve(y_test, p)
        ax.plot(rec, prec, lw=1.6, color=COLORS.get(name), label=result["models"][name]["label"])
    ax.set_xlabel("recall (illicit)")
    ax.set_ylabel("precision (illicit)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    _style(ax)
    ax.legend(fontsize=6.5, loc="lower left", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def comparison_figure(result: dict, path: Path) -> None:
    models = result["models"]
    names = list(models)
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    y = np.arange(len(names))[::-1]
    for yi, n in zip(y, names, strict=True):
        s = models[n]["summary"]
        ci = models[n].get("bootstrap", {}).get("f1_ci95")
        ax.barh(yi, s["f1"]["mean"], color=COLORS.get(n), height=0.6)
        if ci:
            ax.plot(ci, [yi, yi], color="black", lw=1.2)
        ax.text(s["f1"]["mean"] + 0.01, yi + 0.22, f"{s['f1']['mean']:.3f}", fontsize=8)
    ax.set_yticks(y, [models[n]["label"] for n in names], fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("illicit F1, test steps 35-49 (line: 95% time-step bootstrap)")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def make_result_figures(result: dict, curves: dict, y_test: np.ndarray, out: Path) -> None:
    out = Path(out)
    per_step_figure(result, out / "per_step_f1.png")
    pr_figure(result, curves, y_test, out / "pr_curves.png")
    comparison_figure(result, out / "f1_comparison.png")
