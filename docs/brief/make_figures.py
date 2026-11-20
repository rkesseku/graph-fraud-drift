"""Build the brief's figures: an architecture diagram plus copies of the result figures.

Run from the repo root after ``graphfraud run-all``:  python docs/brief/make_figures.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
FIG = HERE / "figures"


def box(ax, x, y, w, h, text, color):
    ax.add_patch(
        FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.03",
                       fc=color, ec="#1f4e9e", lw=1.2)
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.2)


def arrow(ax, x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops={"arrowstyle": "->", "color": "#333333", "lw": 1.1})


def architecture(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 3.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.2)
    ax.axis("off")
    light, mid, warm = "#eef3fb", "#dbe7f7", "#fdebe3"
    data_txt = "Elliptic CSVs\n203k transactions\n234k payment edges\n49 time steps"
    box(ax, 0.1, 1.6, 1.8, 1.1, data_txt, light)
    feat_txt = "Label-free graph features\ndegree, PageRank,\n1- and 2-hop means"
    box(ax, 2.4, 2.75, 2.2, 1.0, feat_txt, mid)
    box(ax, 2.4, 0.45, 2.2, 1.0, "Temporal split\nfit 1-29 | val 30-34\ntest 35-49", mid)
    box(ax, 5.1, 2.75, 2.1, 1.0, "XGBoost x3\nlocal / +aggregated /\n+graph features", light)
    box(ax, 5.1, 1.55, 2.1, 0.95, "GraphSAGE, GAT\n(plain PyTorch,\nfull-batch)", warm)
    # the split and the validation block apply to all five models, so their arrows end on a
    # dashed frame around both model boxes; only XGBoost uses the engineered graph features
    ax.add_patch(FancyBboxPatch((4.95, 1.42), 2.4, 2.46, boxstyle="round,pad=0,rounding_size=0.05",
                                fc="none", ec="#777777", lw=0.9, ls="--"))
    ax.text(6.15, 3.95, "all five models", ha="center", fontsize=7.5, color="#555555")
    box(ax, 5.1, 0.1, 2.1, 0.9, "Validation block\nearly stopping +\ncost threshold", mid)
    box(ax, 7.7, 1.6, 2.2, 1.3, "Test 35-49\nF1, AUPRC, cost,\nper-step drift,\nstep bootstrap CIs",
        light)
    arrow(ax, 1.9, 2.35, 2.4, 3.2)
    arrow(ax, 1.9, 1.9, 2.4, 1.0)
    arrow(ax, 4.6, 3.25, 5.1, 3.25)
    arrow(ax, 4.6, 1.15, 4.95, 1.8)
    arrow(ax, 4.6, 0.8, 5.1, 0.55)
    arrow(ax, 6.15, 1.0, 6.15, 1.42)
    arrow(ax, 7.2, 3.25, 7.7, 2.6)
    arrow(ax, 7.2, 2.0, 7.7, 2.2)
    note = "Refit on steps 1-34 with the\nselected trees / epochs, then\nscore the test steps"
    ax.text(8.8, 1.45, note, ha="center", va="top", fontsize=7.5, color="#555555")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    architecture(FIG / "architecture.png")
    for name in ("per_step_f1.png", "pr_curves.png", "f1_comparison.png"):
        shutil.copy(RESULTS / name, FIG / name)


if __name__ == "__main__":
    main()
