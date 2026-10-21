import csv
from pathlib import Path
import matplotlib.pyplot as plt

plt.rcParams["axes.facecolor"] = "lemonchiffon"
plt.rcParams["figure.facecolor"] = "lightsteelblue"


def plot_convergence():
    # Load our saved experimental results
    path = Path("results/kuhn/cfr_convergence.csv")

    with path.open() as file:
        results = list(csv.DictReader(file))

    iterations = [int(row["iterations"]) for row in results]
    exploitability = [float(row["exploitability"]) for row in results]
    training_times = [float(row["training_seconds"]) for row in results]

    # PLOT 1: How exploitability decreases during training
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(iterations, exploitability, marker="o", label="CFR")
    ax.axhline(0, linestyle="--", color="gray", label="Nash equilibrium")

    ax.set_xscale("log")
    ax.set_title("CFR Convergence — Kuhn Poker")
    ax.set_xlabel("Training iterations")
    ax.set_ylabel("Exploitability")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("results/kuhn/cfr_exploitability.png", dpi=200)
    plt.close(fig)

    # PLOT 2: How training time increases with iterations
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(iterations, training_times, marker="o")

    ax.set_xscale("log")
    ax.set_title("CFR Training Cost — Kuhn Poker")
    ax.set_xlabel("Training iterations")
    ax.set_ylabel("Cumulative training time (seconds)")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig("results/kuhn/cfr_training_time.png", dpi=200)
    plt.close(fig)

    print("Both CFR plots saved successfully!")


if __name__ == "__main__":
    plot_convergence()
