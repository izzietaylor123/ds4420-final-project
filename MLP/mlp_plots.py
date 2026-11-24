"""Figures for the MLP model.

Usage:
  python MLP/mlp_plots.py                       # RMSE and MAE training curves from MLP/mlp_history.json (written by `MLP/mlp_model.py train`)
  python MLP/mlp_plots.py history.json out_dir  # custom input / output directory
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).parent
TRAIN_C, VAL_C = "#2a78d6", "#eb6834"  # categorical slots 1 and 2
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def plot_curves(ep, train, val, baselines, best_epoch, metric, out_path):
    """Train/validation curves per epoch with the baselines as reference lines and the best epoch marked."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)

    # baselines: global mean and movie mean for reference lines
    for name, v in baselines.items():
        ax.axhline(v, color=MUTED, linestyle="--", linewidth=1.2)
        ax.annotate(f"Baseline: predict {name} ({v:.3f})", xy=(ep[0], v), xytext=(0, 4), textcoords="offset points",
                    color=MUTED, fontsize=9, va="bottom")

    ax.plot(ep, train, color=TRAIN_C, linewidth=2, marker="o", markersize=6, markeredgecolor="white",
            markeredgewidth=1.5, label="Train")
    ax.plot(ep, val, color=VAL_C, linewidth=2, marker="o", markersize=6, markeredgecolor="white",
            markeredgewidth=1.5, label="Validation")

    # best epoch: the checkpoint early stopping restores (chosen by validation RMSE)
    bv = val[ep.index(best_epoch)]
    ax.axvline(best_epoch, color=MUTED, linestyle=":", linewidth=1.2)
    ax.annotate(f"Best epoch {best_epoch}\nval {metric} {bv:.3f}", xy=(best_epoch, bv), xytext=(-14, 44),
                textcoords="offset points", color=INK, fontsize=9, ha="right",
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8))

    # direct labels at the line ends (text stays in ink; the mark carries the color)
    ax.text(ep[-1] + 0.15, train[-1], "Train", color=INK, fontsize=10, va="center")
    ax.text(ep[-1] + 0.15, val[-1], "Validation", color=INK, fontsize=10, va="center")

    lo, hi = min(train + val), max(list(baselines.values()) + val)
    ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.12 * (hi - lo))
    ax.set_xlabel("Epoch", color=INK)
    ax.set_ylabel(f"{metric} (rating points)", color=INK)
    ax.set_xticks(ep)
    ax.set_xlim(ep[0] - 0.4, ep[-1] + 1.6)
    ax.tick_params(colors=MUTED)
    ax.set_title(f"MLP training: validation {metric} vs. baselines", color=INK, loc="left", fontsize=13, pad=26)
    ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1, 1.0), ncol=2, labelcolor=INK)
    ax.annotate("Train error is measured with dropout active, so it is slightly pessimistic.",
                xy=(0, -0.16), xycoords="axes fraction", color=MUTED, fontsize=8, ha="left", va="top")

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", out_path)


def plot_training_curves(history_path=HERE / "mlp_history.json", out_dir=HERE):
    h = json.loads(Path(history_path).read_text())
    out_dir = Path(out_dir)
    ep, metric = h["epochs"], h["metric"]
    plot_curves(ep, h["train"], h["val"], h["baselines"], h["best_epoch"], metric, out_dir / "mlp_training_curves.png")
    if h.get("mae", {}).get("train"): # MAE is only tracked for the rating target
        plot_curves(ep, h["mae"]["train"], h["mae"]["val"], h["mae"]["baselines"], h["best_epoch"], "MAE",
                    out_dir / "mlp_training_curves_mae.png")


if __name__ == "__main__":
    plot_training_curves(*sys.argv[1:3])
