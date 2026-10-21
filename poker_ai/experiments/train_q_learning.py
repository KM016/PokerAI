import argparse
import csv
import json
import random
import time
from pathlib import Path

from poker_ai.agents.nash_agent import NashAgent
from poker_ai.agents.strategy_agent import StrategyAgent
from poker_ai.agents.strategy_io import load_strategy, save_strategy
from poker_ai.algorithms.q_learning import QLearningAgent
from poker_ai.evaluation.evaluate import exploitability
from poker_ai.evaluation.matches import balanced_exact_ev
from poker_ai.experiments.agent_tournament import baseline_strategy
from poker_ai.games.kuhn import KuhnPoker


CHECKPOINTS = (
    100, 200, 500,
    1_000, 2_000, 5_000,
    10_000, 20_000, 50_000,
    100_000
)
OPPONENTS = ("Passive", "Random", "Nash", "Mixed")
SCHEDULES = ("Constant", "Decay alpha", "Decay epsilon", "Decay both")


# simulates one hand and updates the q values after each decision made by our agent
def play_training_hand(learner, opponent, learner_seat, rng, visits=None):
    # use a separate rng for reproducible card deals
    game = KuhnPoker()
    game.cards = rng.sample(["J", "Q", "K"], 2)

    while not game.done:
        # if the opponent acts first, wait until the learner gets its turn
        if game.current_player != learner_seat:
            action = opponent.choose_action(
                game.get_information_set(), game.legal_actions()
            )
            game.step(action)
            continue

        # save the information available to the learner before choosing an action
        info_set = game.get_information_set()
        legal_actions = game.legal_actions()
        action = learner.choose_action(info_set, legal_actions)

        # count how often the agent experiences and selects each decision
        if visits is not None:
            if info_set not in visits:
                visits[info_set] = {a: 0 for a in legal_actions}
            visits[info_set][action] += 1

        game.step(action)

        # allow the opponent to respond before updating the earlier decision
        # the next state must be another learner decision or the end of the hand
        while not game.done and game.current_player != learner_seat:
            opponent_action = opponent.choose_action(
                game.get_information_set(), game.legal_actions()
            )
            game.step(opponent_action)

        if game.done:
            # when the hand ends, update using the actual final payoff
            learner.update(
                info_set, action, game.get_payoff()[learner_seat], legal_actions
            )
        else:
            # otherwise estimate the reward using the next learner decision
            # the immediate reward is 0 because the hand has not finished
            learner.update(
                info_set, action, 0, legal_actions,
                game.get_information_set(), game.legal_actions()
            )

    return game.get_payoff()[learner_seat]


# converts learned q values into a valid strategy for exact evaluation and gameplay
def greedy_strategy(learner):
    strategy = {}

    # build a strategy for all 12 possible information sets in Kuhn Poker
    for card in ["J", "Q", "K"]:
        for history in ["", "c", "b", "cb"]:
            info_set = f"{card}|{history}"
            legal_actions = (
                ["check", "bet"] if history in ["", "c"]
                else ["call", "fold"]
            )

            # use equal starting values at information sets not seen during training
            values = learner.q_table.get(
                info_set, {action: 0.0 for action in legal_actions}
            )
            best_value = max(values[action] for action in legal_actions)
            best_actions = [
                action for action in legal_actions
                if values[action] == best_value
            ]

            # choose the highest valued action without training exploration
            # split probability equally if multiple actions have the same q value
            strategy[info_set] = {
                action: (
                    1.0 / len(best_actions) if action in best_actions else 0.0
                )
                for action in legal_actions
            }

    return strategy


# uses the same schedules as our q learning stability experiment
def learning_rates(schedule, hand, total_hands):
    if schedule not in SCHEDULES:
        raise ValueError(f"Unknown schedule: {schedule}")
    if total_hands <= 0 or not 0 <= hand < total_hands:
        raise ValueError("Hand must be within the training range")

    # decay selected rates from 0.1 to 0.01 during the full training run
    progress = hand / max(1, total_hands - 1)
    decayed = 0.1 / (1 + 9 * progress)

    alpha = decayed if schedule in ("Decay alpha", "Decay both") else 0.1
    epsilon = decayed if schedule in ("Decay epsilon", "Decay both") else 0.1
    return alpha, epsilon


# trains and saves a q learning agent against the selected opponent
# default settings are our tuned baseline from the stability experiment
def run_experiment(
    checkpoints=CHECKPOINTS,
    opponent_name="Random",
    schedule="Decay alpha",
    seed=101,
    model_dir="models/kuhn",
    output_dir="results/kuhn"
):
    checkpoints = tuple(checkpoints)
    if not checkpoints or any(h <= 0 for h in checkpoints):
        raise ValueError("Checkpoints must be positive and nonempty")
    if tuple(sorted(set(checkpoints))) != checkpoints:
        raise ValueError("Checkpoints must be strictly increasing")
    if checkpoints[-1] % 2 != 0:
        raise ValueError("Total hands must be even to balance player positions")
    if opponent_name not in OPPONENTS:
        raise ValueError(f"Unknown opponent: {opponent_name}")
    if schedule not in SCHEDULES:
        raise ValueError(f"Unknown schedule: {schedule}")

    # keep the training opponents fixed so only the q learning agent learns
    opponent_strategies = {
        "Passive": baseline_strategy(),
        "Random": baseline_strategy(random_play=True),
        "Nash": NashAgent().strategy
    }
    opponents = {
        name: StrategyAgent(strategy, seed=seed + 10_000 + i)
        for i, (name, strategy) in enumerate(opponent_strategies.items())
    }

    # start a fresh q table and separate rng for reproducible card deals
    learner = QLearningAgent(alpha=0.1, epsilon=0.1, gamma=1.0, seed=seed)
    rng = random.Random(seed + 20_000)
    visits = {}
    results = []
    previous = 0
    training_seconds = 0.0
    total_hands = checkpoints[-1]

    for target in checkpoints:
        # continue training from the previous checkpoint using the same q table
        start = time.perf_counter()
        for hand in range(previous, target):
            # adjust alpha and epsilon using the chosen learning schedule
            learner.alpha, learner.epsilon = learning_rates(
                schedule, hand, total_hands
            )

            # mixed training randomly selects a different opponent each hand
            current_opponent = (
                rng.choice(("Passive", "Random", "Nash"))
                if opponent_name == "Mixed" else opponent_name
            )

            # alternate seats so both player positions receive experience
            play_training_hand(
                learner, opponents[current_opponent], hand % 2, rng, visits
            )
        training_seconds += time.perf_counter() - start

        # convert q values into a policy with no random exploration
        strategy = greedy_strategy(learner)

        # measure exact profit against every opponent, not just the training one
        ev_passive = balanced_exact_ev(strategy, opponent_strategies["Passive"])
        ev_random = balanced_exact_ev(strategy, opponent_strategies["Random"])
        ev_nash = balanced_exact_ev(strategy, opponent_strategies["Nash"])
        vulnerability = exploitability(strategy)
        coverage = sum(sum(counts.values()) > 0 for counts in visits.values())

        results.append({
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

    # include the opponent and schedule in filenames to preserve previous models
    filename = f"q_learning_{opponent_name.lower()}_{schedule.lower().replace(' ', '_')}"
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    policy_path = model_dir / f"{filename}.json"
    values_path = model_dir / f"{filename}_q_values.json"
    metadata_path = model_dir / f"{filename}_metadata.json"
    csv_path = output_dir / f"{filename}.csv"

    # save the playable policy and original q values separately
    save_strategy(strategy, policy_path)
    save_strategy(learner.q_table, values_path)

    # record training settings and results so this model is reproducible
    metadata = {
        "algorithm": "tabular_q_learning",
        "game": "kuhn",
        "training_opponent": opponent_name,
        "schedule": schedule,
        "training_hands": total_hands,
        "checkpoints": list(checkpoints),
        "seed": seed,
        "initial_alpha": 0.1,
        "final_alpha": learner.alpha,
        "initial_epsilon": 0.1,
        "final_epsilon": learner.epsilon,
        "gamma": 1.0,
        "training_seconds": training_seconds,
        "evaluation": results[-1]
    }
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4, allow_nan=False)

    # save the results at every checkpoint for future plots and comparisons
    output_dir.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    # confirm the saved strategy contains exactly the policy we evaluated
    assert load_strategy(policy_path) == strategy

    print(f"\nTraining opponent: {opponent_name}")
    print(f"Learning schedule: {schedule}")
    print(f"Training time: {training_seconds:.3f}s")
    print(f"Policy saved to: {policy_path}")
    print(f"Q values saved to: {values_path}")
    print(f"Metadata saved to: {metadata_path}")
    print(f"Results saved to: {csv_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Kuhn Poker Q-learning agent")
    parser.add_argument("--opponent", choices=OPPONENTS, default="Random")
    parser.add_argument("--schedule", choices=SCHEDULES, default="Decay alpha")
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()
    run_experiment(opponent_name=args.opponent, schedule=args.schedule, seed=args.seed)
