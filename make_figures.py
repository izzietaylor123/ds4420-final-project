"""Build the report's table and figure for the CNN method.

  results/table1_rmse.png   prediction error: past-ratings baselines, our CNN, shuffled control
  results/fig_history.png   prediction error vs. how many movies the user has rated

Run after: python poster_cnn.py, then python poster_model.py evaluate
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS = Path(__file__).parent / "results"
INK, MUTED, GRID, BLUE = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"


def load(name):
    with open(RESULTS / f"eval_{name}.json") as f:
        return json.load(f)


def table1_png():
    """Results table as an image, to screenshot into the report."""
    cnn = load("posters_cnn")
    base = cnn["baselines"]
    b = base["bias"]
    rows = [("Everyone's average (not personalized)", base["global_mean"]),
            ("User's own average", base["user_mean"]),
            ("User + movie tendencies (baseline)", b),
            ("Baseline + similar posters (our CNN)", cnn["model"]["test_rmse"]),
            ("Control: shuffled posters", cnn["shuffled"]["test_rmse"])]
    cells = [[n, f"{v:.3f}", "" if i == 2 else f"{v - b:+.3f}"] for i, (n, v) in enumerate(rows)]
    fig, ax = plt.subplots(figsize=(6.4, 2.3), dpi=200)
    ax.axis("off")
    tbl = ax.table(cellText=cells, colLabels=["Prediction method", "Error (stars)", "vs. baseline"],
                   colWidths=[0.62, 0.19, 0.19], loc="center", cellLoc="left", colLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_text_props(color=INK, weight="bold" if r in (0, 4) else "normal")
        if r == 0:
            cell.set_facecolor("#f4f3f0")
    fig.text(0.02, 0.03, "Error = RMSE on 288,231 held-out ratings: how many stars predictions are typically off (lower is better).",
             fontsize=7.5, color=MUTED)
    fig.savefig(RESULTS / "table1_rmse.png", bbox_inches="tight")
    print("saved results/table1_rmse.png")


def history_figure():
    """Prediction error grouped by how many ratings the user has in training."""
    import numpy as np
    from poster_model import PosterSimilarity, load_features, load_ratings, rmse, split

    cnn = load("posters_cnn")
    d = load_ratings()
    df, nu = d["ratings"], len(d["user_ids"])
    u, m, r = df["u"].to_numpy(), df["m"].to_numpy(), df["rating"].to_numpy().astype(float)
    sp = split(len(df))
    tr, te = sp["train"], sp["test"]
    F, has = load_features(d["movie_ids"], "poster_features_cnn.npz")
    model = PosterSimilarity(F, has, u[tr], m[tr], r[tr], nu)
    pred = model.predict(u[te], m[te], grid=[(cnn["model"]["k"], cnn["model"]["lam"])])[:, 0]
    base = np.clip(model.baseline(u[te], m[te]), 0.5, 5)
    hist = np.bincount(u[tr], minlength=nu)[u[te]]

    edges, labels = [0, 25, 50, 100, 200, 400, 10 ** 9], ["<25", "25-50", "50-100", "100-200", "200-400", "400+"]
    rows = []
    for lo, hi, lab in zip(edges[:-1], edges[1:], labels):
        s_ = (hist >= lo) & (hist < hi)
        rows.append({"bucket": lab, "n_ratings": int(s_.sum()), "n_users": int(len(np.unique(u[te][s_]))),
                     "baseline": rmse(base[s_], r[te][s_]), "cnn": rmse(pred[s_], r[te][s_])})
    with open(RESULTS / "history_buckets.json", "w") as f:
        json.dump(rows, f, indent=2)

    fig, ax = plt.subplots(figsize=(6.4, 3.8), dpi=200)
    x = range(len(rows))
    yb = [row["baseline"] for row in rows]
    yc = [row["cnn"] for row in rows]
    ax.plot(x, yb, color=MUTED, linewidth=2, linestyle="--", marker="o", markersize=5, label="Baseline (no posters)")
    ax.plot(x, yc, color=BLUE, linewidth=2, marker="o", markersize=6, markeredgecolor="white",
            markeredgewidth=1.5, label="Baseline + our poster CNN")
    ax.set_xticks(list(x), [f"{row['bucket']}\n({row['n_users']:,} users)" for row in rows], fontsize=8)
    ax.set_xlim(-0.3, len(rows) - 1 + 0.3)
    ax.set_xlabel("Number of movies the user has rated", fontsize=9, color=MUTED)
    ax.set_ylabel("Prediction error (RMSE, stars)", fontsize=9, color=MUTED)
    fig.suptitle("Our poster CNN lowers prediction error for every group of users", x=0.02, ha="left",
                 fontsize=10.5, color=INK)
    fig.text(0.02, 0.905, "Groups are based on how many movies each user has rated (lower is better).",
             fontsize=8.5, color=MUTED, ha="left")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for sp_ in ("top", "right", "left"):
        ax.spines[sp_].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="both", length=0, colors=MUTED, labelsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(RESULTS / "fig_history.png")
    print("saved results/fig_history.png")


if __name__ == "__main__":
    table1_png()
    history_figure()
