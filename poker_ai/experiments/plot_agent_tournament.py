import csv
from pathlib import Path

import matplotlib.pyplot as plt


plt.rcParams.update({
    "axes.facecolor": "lemonchiffon",
    "figure.facecolor": "lightsteelblue",
    "savefig.facecolor": "auto",
})


def plot_tournament():
    path = Path("results/kuhn/agent_tournament.csv")

    with path.open(newline="") as file:
        results = list(csv.DictReader(file))

    opponents = [row["opponent"] for row in results]
    simulated = [float(row["simulated_ev"]) for row in results]
    exact = [float(row["exact_ev"]) for row in results]
    win_rates = [float(row["win_rate"]) for row in results]

    positions = list(range(len(opponents)))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.bar(
        [x - width / 2 for x in positions], simulated,
        width, color="lightsteelblue", edgecolor="#34465d",
        label="Simulated EV"
    )
    ax.bar(
        [x + width / 2 for x in positions], exact,
        width, color="#526e95", edgecolor="#34465d",
        label="Exact EV"
    )

    ax.axhline(0, color="#34465d", linewidth=0.9)
    ax.set_xticks(
        positions,
        [f"{name}\nWin rate: {rate:.1%}" for name, rate in zip(opponents, win_rates)]
    )
    ax.set_title("CFR+ Tournament: Simulated vs Exact Payoff")
    ax.set_ylabel("Expected payoff per hand")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend()

    fig.tight_layout()
    fig.savefig("results/kuhn/agent_tournament.png", dpi=200)
    plt.close(fig)

    print("Tournament plot saved successfully!")


if __name__ == "__main__":
    plot_tournament()
