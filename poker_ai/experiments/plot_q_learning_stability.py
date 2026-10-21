import csv
from pathlib import Path

import matplotlib.pyplot as plt


# use the same background colours as the other poker research figures
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

COLOURS = {
    "Constant": "#21334c",
    "Decay alpha": "#247c83",
    "Decay epsilon": "#965f40",
    "Decay both": "#734f92"
}


def plot_stability(output_dir="results/kuhn"):
    # read saved mean and standard deviation without repeating training
    output_dir = Path(output_dir)
    with (output_dir / "q_learning_stability_summary.csv").open() as file:
        results = list(csv.DictReader(file))

    plots = [
        ("ev_mean", "ev_sd", "EV vs Random", "q_learning_stability_ev.png"),
        ("exploitability_mean", "exploitability_sd", "Exploitability", "q_learning_stability_exploitability.png")
    ]

    for value_key, sd_key, y_label, filename in plots:
        fig, ax = plt.subplots(figsize=(9, 5))

        for name, colour in COLOURS.items():
            data = [row for row in results if row["configuration"] == name]
            if not data:
                continue

            hands = [int(row["hands"]) for row in data]
            values = [float(row[value_key]) for row in data]
            sds = [float(row[sd_key]) for row in data]

            # plot mean performance and shade one sample standard deviation
            ax.plot(hands, values, marker="o", markersize=3, linewidth=2,
                    label=name, color=colour)
            ax.fill_between(
                hands,
                [value - sd for value, sd in zip(values, sds)],
                [value + sd for value, sd in zip(values, sds)],
                color=colour,
                alpha=0.12
            )

        ax.set_xscale("log")
        ax.set_xlabel("Training hands")
        ax.set_ylabel(y_label)
        ax.set_title(f"Q-learning stability: {y_label}")
        ax.grid(alpha=0.2)
        ax.legend()

        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=200)
        plt.close(fig)

    print(f"Stability plots saved to: {output_dir}")


if __name__ == "__main__":
    plot_stability()
