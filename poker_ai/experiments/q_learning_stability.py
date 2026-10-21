import csv
import random
from pathlib import Path
from statistics import mean, stdev

from poker_ai.agents.nash_agent import NashAgent
from poker_ai.agents.strategy_agent import StrategyAgent
from poker_ai.algorithms.q_learning import QLearningAgent
from poker_ai.evaluation.evaluate import exploitability
from poker_ai.evaluation.matches import balanced_exact_ev
from poker_ai.experiments.agent_tournament import baseline_strategy
from poker_ai.experiments.train_q_learning import greedy_strategy, play_training_hand


# compares whether decaying alpha or epsilon makes q learning more stable
# use several random seeds so we do not base conclusions on one training run
CONFIGURATIONS = (
    "Constant",
    "Decay alpha",
    "Decay epsilon",
    "Decay both"
)

CHECKPOINTS = (
    100, 200, 500,
    1_000, 2_000, 5_000,
    10_000, 20_000, 50_000,
    100_000
)


def learning_rates(configuration, hand, total_hands):
    # reduce a parameter from 0.1 towards 0.01 over the full training run
    # a gradual reduction should make late updates less sensitive to random hands
    progress = hand / max(1, total_hands - 1)
    decayed = 0.1 / (1 + 9 * progress)

    # change only the parameters specified by the selected configuration
    alpha = decayed if configuration in ("Decay alpha", "Decay both") else 0.1
    epsilon = decayed if configuration in ("Decay epsilon", "Decay both") else 0.1
    return alpha, epsilon


def save_rows(path, rows, columns):
    # create the results folder and write the data so plots can be reproduced
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(
    checkpoints=CHECKPOINTS,
    seeds=(42, 43, 44, 45, 46),
    configurations=CONFIGURATIONS,
    output_dir="results/kuhn",
    verbose=True
):
    # compare the same four settings over the same number of training hands
    checkpoints = tuple(checkpoints)
    seeds = tuple(seeds)
    configurations = tuple(configurations)

    if not checkpoints or any(h <= 0 for h in checkpoints):
        raise ValueError("Checkpoints must be positive and nonempty")
    if tuple(sorted(set(checkpoints))) != checkpoints:
        raise ValueError("Checkpoints must be strictly increasing")
    if not seeds:
        raise ValueError("At least one seed is required")
    if not configurations or any(c not in CONFIGURATIONS for c in configurations):
        raise ValueError("Unknown or missing configuration")

    # random bets and calls, so the agent can experience all 12 information sets
    # nash and passive are kept as additional opponents for exact evaluation
    random_strategy = baseline_strategy(random_play=True)
    passive_strategy = baseline_strategy()
    nash_strategy = NashAgent().strategy
    total_hands = checkpoints[-1]
    runs = []

    for configuration in configurations:
        if verbose:
            print(f"\n--- {configuration} ---")

        for seed in seeds:
            # start with a fresh q table for each seed and configuration
            # use the same seeds across configurations for a fair comparison
            learner = QLearningAgent(alpha=0.1, epsilon=0.1, gamma=1.0, seed=seed)
            opponent = StrategyAgent(random_strategy, seed=seed + 10_000)
            rng = random.Random(seed + 20_000)
            visits = {}
            previous = 0

            for target in checkpoints:
                # continue the same training run rather than restarting at checkpoints
                for hand in range(previous, target):
                    learner.alpha, learner.epsilon = learning_rates(
                        configuration, hand, total_hands
                    )

                    # alternate player positions and record experienced information sets
                    play_training_hand(learner, opponent, hand % 2, rng, visits)

                # remove exploration when converting q values into a policy
                strategy = greedy_strategy(learner)

                # exact EV measures profit against a known opponent without simulation noise
                # exploitability measures vulnerability to a perfect best response
                row = {
                    "configuration": configuration,
                    "seed": seed,
                    "hands": target,
                    "alpha": learner.alpha,
                    "epsilon": learner.epsilon,
                    "ev_vs_random": balanced_exact_ev(strategy, random_strategy),
                    "ev_vs_passive": balanced_exact_ev(strategy, passive_strategy),
                    "ev_vs_nash": balanced_exact_ev(strategy, nash_strategy),
                    "exploitability": exploitability(strategy),
                    "visited_info_sets": sum(
                        sum(counts.values()) > 0 for counts in visits.values()
                    )
                }
                runs.append(row)
                previous = target

            if verbose:
                print(
                    f"Seed {seed} | "
                    f"Final EV vs Random: {row['ev_vs_random']:+.4f} | "
                    f"Exploitability: {row['exploitability']:.4f}"
                )

    # calculate mean and sample standard deviation across seeds at each checkpoint
    # a smaller spread indicates that training depends less on the random seed
    summary = []
    for configuration in configurations:
        for target in checkpoints:
            sample = [
                row for row in runs
                if row["configuration"] == configuration and row["hands"] == target
            ]
            ev = [row["ev_vs_random"] for row in sample]
            risk = [row["exploitability"] for row in sample]

            summary.append({
                "configuration": configuration,
                "hands": target,
                "seeds": len(sample),
                "ev_mean": mean(ev),
                "ev_sd": stdev(ev) if len(ev) > 1 else 0.0,
                "exploitability_mean": mean(risk),
                "exploitability_sd": stdev(risk) if len(risk) > 1 else 0.0,
                "coverage_mean": mean(row["visited_info_sets"] for row in sample)
            })

    output_dir = Path(output_dir)
    save_rows(
        output_dir / "q_learning_stability_runs.csv", runs,
        ["configuration", "seed", "hands", "alpha", "epsilon",
         "ev_vs_random", "ev_vs_passive", "ev_vs_nash",
         "exploitability", "visited_info_sets"]
    )
    save_rows(
        output_dir / "q_learning_stability_summary.csv", summary,
        ["configuration", "hands", "seeds", "ev_mean", "ev_sd",
         "exploitability_mean", "exploitability_sd", "coverage_mean"]
    )

    if verbose:
        print("\n--- Mean results at the final checkpoint ---")
        for row in summary:
            if row["hands"] == total_hands:
                print(
                    f"{row['configuration']}: "
                    f"EV {row['ev_mean']:+.4f} ± {row['ev_sd']:.4f} | "
                    f"Exploitability {row['exploitability_mean']:.4f} "
                    f"± {row['exploitability_sd']:.4f}"
                )
        print(f"\nResults saved to: {output_dir}")

    return runs, summary


if __name__ == "__main__":
    run_experiment()
