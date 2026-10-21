import csv
from pathlib import Path

import matplotlib.pyplot as plt


# reuse the background colours from our other kuhn poker experiments
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
    "Baseline": "#21334c",
    "More exploration": "#247c83",
    "Lower learning rate": "#8c5544",
    "Squared error loss": "#725493"
}
PLOTS = (
    ("q_call_minus_fold", "Predicted Q(call) - Q(fold) at Q|b", "dqn_qb_q_gap.png"),
    ("ev_vs_mixed", "Exact EV vs Mixed", "dqn_qb_ev_mixed.png"),
    ("q_call_rate", "Fraction of Q|b visits choosing call", "dqn_qb_call_rate.png")
)


def plot_diagnostic(output_dir="results/kuhn"):
    output_dir = Path(output_dir)
    with (output_dir / "dqn_qb_diagnostic_summary.csv").open(encoding="utf-8") as file:
        summary = list(csv.DictReader(file))

    for metric, ylabel, filename in PLOTS:
        fig, ax = plt.subplots(figsize=(9, 5))
        if metric == "q_call_minus_fold":
            ax.axhline(0, color="#616161", linestyle="--", linewidth=1.25, alpha=0.75)
        for configuration, colour in COLOURS.items():
            rows = [row for row in summary if row["configuration"] == configuration]
            if not rows:
                continue
            rows = [row for row in rows if row[f"{metric}_mean"]]
            if not rows:
                continue
            hands = [int(r["hands"]) for r in rows]
            means = [float(r[f"{metric}_mean"]) for r in rows]
            sds = [float(r[f"{metric}_sd"]) for r in rows]
            ax.plot(hands, means, marker="o", markersize=4, linewidth=2,
                    color=colour, label=configuration)
            ax.fill_between(hands, [v-s for v, s in zip(means, sds)],
                            [v+s for v, s in zip(means, sds)],
                            color=colour, alpha=0.13)

        ax.set_xscale("log")
        ax.set_xlabel("Training hands")
        ax.set_ylabel(ylabel)
        ax.set_title(f"DQN Q|b diagnostic: {ylabel}")
        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=200)
        plt.close(fig)

    print(f"Diagnostic plots saved to: {output_dir}")


if __name__ == "__main__":
    plot_diagnostic()
