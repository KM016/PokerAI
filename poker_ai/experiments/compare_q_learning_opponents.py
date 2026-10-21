import csv
import random
from pathlib import Path

from poker_ai.agents.nash_agent import NashAgent
from poker_ai.agents.strategy_agent import StrategyAgent
from poker_ai.agents.strategy_io import save_strategy
from poker_ai.algorithms.q_learning import QLearningAgent
from poker_ai.evaluation.evaluate import exploitability
from poker_ai.evaluation.matches import balanced_exact_ev
from poker_ai.experiments.agent_tournament import baseline_strategy
from poker_ai.experiments.train_q_learning import play_training_hand, greedy_strategy


# compares q learning after training against different fixed opponents
# each agent learns independently so we can compare the effect of training data
def run_comparison(
    checkpoints=(
        100, 200, 500,
        1_000, 2_000, 5_000,
        10_000, 20_000, 50_000,
        100_000
    ),
    seed=42
):
    # these are the known strategies used for training and evaluation
    opponent_strategies = {
        "Passive": baseline_strategy(),
        "Random": baseline_strategy(random_play=True),
        "Nash": NashAgent().strategy
    }

    # mixed chooses a different opponent for each training hand
    training_modes = ["Passive", "Random", "Nash", "Mixed"]
    results = []

    for i, mode in enumerate(training_modes):
        print(f"\n--- Training against {mode} ---")

        # start from a new q table for each training mode
        learner = QLearningAgent(alpha=0.1, epsilon=0.1, gamma=1.0, seed=seed + 100 * i)

        # use independent random generators for card deals and opponent actions
        rng = random.Random(seed + 100 * i + 1)
        opponents = {
            name: StrategyAgent(strategy, seed=seed + 100 * i + 10 + j)
            for j, (name, strategy) in enumerate(opponent_strategies.items())
        }

        # record which information sets our agent experienced during training
        visits = {}
        previous = 0

        for target in checkpoints:
            # continue training without resetting the learned q values
            for hand in range(previous, target):
                # mixed samples uniformly from the three opponent strategies
                opponent_name = (
                    rng.choice(("Passive", "Random", "Nash"))
                    if mode == "Mixed" else mode
                )

                # alternate seats so both player positions are learned
                play_training_hand(
                    learner, opponents[opponent_name], hand % 2, rng, visits
                )

            # convert q values into action probabilities for exact evaluation
            strategy = greedy_strategy(learner)

            # measure profitability against every opponent, not just the training one
            ev_passive = balanced_exact_ev(strategy, opponent_strategies["Passive"])
            ev_random = balanced_exact_ev(strategy, opponent_strategies["Random"])
            ev_nash = balanced_exact_ev(strategy, opponent_strategies["Nash"])

            # evaluate robustness and how much of the game was experienced
            vulnerability = exploitability(strategy)
            coverage = sum(sum(counts.values()) > 0 for counts in visits.values())

            results.append({
                "training_opponent": mode,
                "hands": target,
                "ev_vs_passive": ev_passive,
                "ev_vs_random": ev_random,
                "ev_vs_nash": ev_nash,
                "exploitability": vulnerability,
                "visited_info_sets": coverage
            })

            print(
                f"{target:,} hands | "
                f"vs Passive: {ev_passive:+.4f} | "
                f"vs Random: {ev_random:+.4f} | "
                f"vs Nash: {ev_nash:+.4f} | "
                f"Exploitability: {vulnerability:.4f} | "
                f"Visited: {coverage}/12"
            )
            previous = target

        # save each trained policy separately for later matches and comparisons
        save_strategy(strategy, f"models/kuhn/q_learning_{mode.lower()}.json")

    # store all checkpoint results together so we can plot learning curves
    output = Path("results/kuhn/q_learning_opponents.csv")
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "training_opponent", "hands", "ev_vs_passive",
            "ev_vs_random", "ev_vs_nash", "exploitability",
            "visited_info_sets"
        ])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {output}")
    return results


if __name__ == "__main__":
    run_comparison()
