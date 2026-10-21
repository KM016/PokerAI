import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from poker_ai.experiments.analyse_dqn_strategies import INFO_SETS


# use the same research plot colours as the other poker experiments
plt.rcParams.update({
    "axes.facecolor": "lemonchiffon",
    "figure.facecolor": "lightsteelblue"
})


def plot_strategies(output_dir="results/kuhn"):
    output_dir = Path(output_dir)
    with (output_dir / "dqn_strategy_actions.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError("No strategy results to plot")

    seeds = list(dict.fromkeys(int(row["seed"]) for row in rows))
    labels = [f"R{seed}" if opponent == "Random" else f"M{seed}"
              for seed in seeds for opponent in ("Random", "Mixed")]
    actions = ("check", "bet", "call", "fold")
    lookup = {(int(row["seed"]), row["info_set"]): row for row in rows}
    matrix = np.zeros((len(INFO_SETS), len(labels)), dtype=int)

    for i, info in enumerate(INFO_SETS):
        for j, seed in enumerate(seeds):
            if (seed, info) not in lookup:
                raise ValueError(f"Missing strategy result for seed {seed}, {info}")
            for k, key in enumerate(("random_action", "mixed_action")):
                action = lookup[(seed, info)][key]
                if action not in actions:
                    raise ValueError(f"Unknown action: {action}")
                matrix[i, 2 * j + k] = actions.index(action)

    colours = ["#B6CADF", "#CCA5D9", "#A6D7C2", "#E8BD89"]
    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.1), 8))
    ax.imshow(matrix, cmap=ListedColormap(colours), vmin=-0.5, vmax=3.5,
              interpolation="nearest", aspect="auto")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(INFO_SETS)), INFO_SETS)
    ax.set_xlabel("Training opponent and seed   R = Random, M = Mixed")
    ax.set_ylabel("Kuhn Poker information set")
    ax.set_title("DQN learned actions across seeds")
    ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(INFO_SETS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for i in range(len(INFO_SETS)):
        for j in range(len(labels)):
            ax.text(j, i, actions[matrix[i, j]], ha="center", va="center", fontsize=8,
                    color="#233449")
    ax.legend(
        handles=[Patch(facecolor=colours[i], label=actions[i]) for i in range(4)],
        loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False
    )
    fig.tight_layout()
    path = output_dir / "dqn_strategy_actions.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot saved DQN poker action choices")
    parser.add_argument("--output-dir", default="results/kuhn")
    args = parser.parse_args()
    print(f"Plot saved to: {plot_strategies(args.output_dir)}")
