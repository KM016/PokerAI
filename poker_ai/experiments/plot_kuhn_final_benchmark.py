import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# keep the same colours used in our earlier research figures
plt.rcParams.update({
    "figure.facecolor": "lightsteelblue",
    "axes.facecolor": "lemonchiffon",
    "font.size": 11
})


def plot_benchmark(output_dir="results/kuhn"):
    output = Path(output_dir)
    with (output / "kuhn_final_benchmark.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError("Benchmark results are empty")

    labels = [row["agent"] for row in rows]
    position = np.arange(len(rows))
    width = 0.35

    # profitability is a different objective from robustness
    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.bar(position - width / 2, [float(row["ev_vs_random"]) for row in rows],
           width, label="EV vs Random", color="#263a56")
    ax.bar(position + width / 2, [float(row["ev_vs_mixed"]) for row in rows],
           width, label="EV vs Mixed", color="#287c86")
    ax.axhline(0, color="#222222", linewidth=0.8)
    ax.set_ylabel("Exact expected payoff per hand")
    ax.set_title("Kuhn Poker: Profitability against fixed opponents")
    ax.set_xticks(position, labels, rotation=30, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "kuhn_final_profitability.png", dpi=170)
    plt.close(fig)

    # exploitability measures performance against a best responding opponent
    fig, ax = plt.subplots(figsize=(12, 6.5))
    bars = ax.bar(position, [float(row["exploitability"]) for row in rows],
                  color="#66518d")
    ax.set_ylabel("Exploitability (lower is better)")
    ax.set_title("Kuhn Poker: Strategy robustness")
    ax.set_xticks(position, labels, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.2)
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.4f}", (bar.get_x() + bar.get_width() / 2, height),
                    ha="center", va="bottom", fontsize=8, xytext=(0, 3),
                    textcoords="offset points")
    ax.set_ylim(bottom=0, top=max(float(row["exploitability"]) for row in rows) * 1.18 + 0.01)
    fig.tight_layout()
    fig.savefig(output / "kuhn_final_exploitability.png", dpi=170)
    plt.close(fig)
    print(f"Benchmark plots saved to: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot the final Kuhn Poker benchmark")
    parser.add_argument("--output-dir", default="results/kuhn")
    args = parser.parse_args()
    plot_benchmark(args.output_dir)
