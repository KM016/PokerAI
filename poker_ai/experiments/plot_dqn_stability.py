import csv
from pathlib import Path

import matplotlib.pyplot as plt


# match the backgrounds used for the other poker research figures
plt.rcParams.update({
    "axes.facecolor": "lemonchiffon",
    "figure.facecolor": "lightsteelblue",
    "savefig.facecolor": "auto",
    "text.color": "#21334c",
    "axes.labelcolor": "#21334c",
    "axes.edgecolor": "#21334c",
    "xtick.color": "#21334c",
    "ytick.color": "#21334c"
})

COLOURS = {"Random": "#21334c", "Mixed": "#247c83"}
PLOTS = (
    ("ev_vs_mixed", "Exact EV vs Mixed", "dqn_stability_ev_mixed.png"),
    ("ev_vs_random", "Exact EV vs Random", "dqn_stability_ev_random.png"),
    ("exploitability", "Exploitability", "dqn_stability_exploitability.png")
)


def plot_stability(output_dir="results/kuhn"):
    # use the saved summaries so plotting never repeats the neural network training
    output_dir = Path(output_dir)
    with (output_dir / "dqn_stability_summary.csv").open(encoding="utf-8") as file:
        summary = list(csv.DictReader(file))

    for metric, ylabel, filename in PLOTS:
        fig, ax = plt.subplots(figsize=(9, 5))
        for opponent, colour in COLOURS.items():
            rows = [row for row in summary if row["training_opponent"] == opponent]
            if not rows:
                continue

            hands = [int(row["hands"]) for row in rows]
            values = [float(row[f"{metric}_mean"]) for row in rows]
            sds = [float(row[f"{metric}_sd"]) for row in rows]

            # compare the mean learning curve and one sample standard deviation
            ax.plot(hands, values, marker="o", markersize=3,
                    linewidth=2, label=f"Trained on {opponent}", color=colour)
            ax.fill_between(
                hands,
                [value - sd for value, sd in zip(values, sds)],
                [value + sd for value, sd in zip(values, sds)],
                color=colour, alpha=0.13
            )

        ax.set_xscale("log")
        ax.set_xlabel("Training hands")
        ax.set_ylabel(ylabel)
        ax.set_title(f"DQN stability across seeds: {ylabel}")
        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=200)
        plt.close(fig)

    print(f"DQN stability plots saved to: {output_dir}")


if __name__ == "__main__":
    plot_stability()
