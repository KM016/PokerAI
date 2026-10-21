import csv
from pathlib import Path

import matplotlib.pyplot as plt


plt.rcParams.update({
    "axes.facecolor": "lemonchiffon",
    "figure.facecolor": "lightsteelblue",
    "savefig.facecolor": "auto",
})


def plot_ablation():
    path = Path("results/kuhn/cfr_ablation.csv")

    with path.open(newline="") as file:
        results = list(csv.DictReader(file))

    names = [row["configuration"] for row in results]
    scores = [float(row["exploitability"]) for row in results]

    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.barh(
        names, scores, left=1e-7,
        color="lightsteelblue", edgecolor="#34465d", linewidth=1.2
    )

    for bar, score in zip(bars, scores):
        ax.text(
            score * 1.25,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.8f}",
            va="center", fontsize=10, color="#24364d"
        )

    ax.set_xscale("log")
    ax.set_xlim(1e-7, 1e-2)
    ax.invert_yaxis()
    ax.set_title("CFR+ Sequential Component Analysis")
    ax.set_xlabel("Exploitability (log scale, lower is better)")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig("results/kuhn/cfr_ablation.png", dpi=200)
    plt.close(fig)

    print("Ablation plot saved successfully!")


if __name__ == "__main__":
    plot_ablation()
