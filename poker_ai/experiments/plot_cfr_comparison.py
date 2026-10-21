import csv
from pathlib import Path
import matplotlib.pyplot as plt

plt.rcParams["axes.facecolor"] = "lemonchiffon"
plt.rcParams["figure.facecolor"] = "lightsteelblue"


def plot_comparison():
    # Load results from our benchmark
    path = Path("results/kuhn/cfr_comparison.csv")

    with path.open() as file:
        results = list(csv.DictReader(file))

    algorithms = ["CFR", "CFR+"]

    # PLOT 1: Exploitability vs training iterations
    fig, ax = plt.subplots(figsize=(8, 5))

    for algorithm in algorithms:
        data = [r for r in results if r["algorithm"] == algorithm]

        iterations = [int(r["iterations"]) for r in data]
        scores = [float(r["exploitability"]) for r in data]

        ax.plot(iterations, scores, marker="o", label=algorithm)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("CFR vs CFR+ — Convergence")
    ax.set_xlabel("Training iterations")
    ax.set_ylabel("Exploitability")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig("results/kuhn/cfr_comparison_iterations.png", dpi=200)
    plt.close(fig)

    # PLOT 2: Exploitability vs training time
    fig, ax = plt.subplots(figsize=(8, 5))

    for algorithm in algorithms:
        data = [r for r in results if r["algorithm"] == algorithm]

        times = [float(r["training_seconds"]) for r in data]
        scores = [float(r["exploitability"]) for r in data]

        ax.plot(times, scores, marker="o", label=algorithm)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("CFR vs CFR+ — Computational Efficiency")
    ax.set_xlabel("Cumulative training time (seconds)")
    ax.set_ylabel("Exploitability")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig("results/kuhn/cfr_comparison_time.png", dpi=200)
    plt.close(fig)

    print("CFR comparison plots saved successfully!")


if __name__ == "__main__":
    plot_comparison()
