import argparse
import csv
import json
import random
import time
from pathlib import Path

import torch

from poker_ai.agents.dqn_agent import DQNAgent
from poker_ai.agents.nash_agent import NashAgent
from poker_ai.agents.strategy_agent import StrategyAgent
from poker_ai.agents.strategy_io import load_strategy, save_strategy
from poker_ai.algorithms.dqn import CARDS, HISTORIES, encode_info_set
from poker_ai.evaluation.evaluate import exploitability
from poker_ai.evaluation.matches import balanced_exact_ev
from poker_ai.experiments.agent_tournament import baseline_strategy
from poker_ai.experiments.train_q_learning import CHECKPOINTS, OPPONENTS
from poker_ai.games.kuhn import KuhnPoker


# stores each decision as a transition to the next learner decision or a terminal payoff
def play_training_hand(learner, opponent, learner_seat, rng, visits=None):
    if learner_seat not in (0, 1):
        raise ValueError("Learner seat must be 0 or 1")

    game = KuhnPoker()
    game.cards = rng.sample(list(CARDS), 2)

    while not game.done:
        # allow the opponent to act before the learner if the opponent has the first move
        if game.current_player != learner_seat:
            action = opponent.choose_action(
                game.get_information_set(), game.legal_actions()
            )
            game.step(action)
            continue

        info_set = game.get_information_set()
        legal_actions = game.legal_actions()
        action = learner.choose_action(info_set, legal_actions)

        if visits is not None:
            if info_set not in visits:
                visits[info_set] = {choice: 0 for choice in legal_actions}
            visits[info_set][action] += 1

        game.step(action)

        # keep going until the learner acts again or the hand finishes
        # this is important because the opponent move is part of the transition
        while not game.done and game.current_player != learner_seat:
            opponent_action = opponent.choose_action(
                game.get_information_set(), game.legal_actions()
            )
            game.step(opponent_action)

        if game.done:
            # the completed hand gives the actual reward in the learner seat
            learner.remember(
                info_set, action, game.get_payoff()[learner_seat], legal_actions
            )
        else:
            # an ongoing hand has zero immediate reward and a future learner decision
            learner.remember(
                info_set, action, 0.0, legal_actions,
                game.get_information_set(), game.legal_actions()
            )

    return game.get_payoff()[learner_seat]


# turn neural q predictions into a deterministic policy for exact evaluation
def greedy_strategy(learner):
    strategy = {}

    # disable gradient tracking and exploration when extracting the final policy
    with torch.no_grad():
        for card in CARDS:
            for history in HISTORIES:
                info_set = f"{card}|{history}"
                actions = (
                    ("check", "bet") if history in ("", "c")
                    else ("call", "fold")
                )
                values = learner.policy_net(encode_info_set(info_set))
                best_action = int(torch.argmax(values).item())
                strategy[info_set] = {
                    action: float(i == best_action)
                    for i, action in enumerate(actions)
                }

    return strategy


# train the same dqn implementation against different fixed opponent distributions
def run_experiment(
    checkpoints=CHECKPOINTS,
    opponent_name="Random",
    seed=101,
    learning_rate=0.001,
    epsilon=0.1,
    batch_size=32,
    replay_capacity=10_000,
    target_update=100,
    learn_every=16,
    model_dir="models/kuhn",
    output_dir="results/kuhn",
    loss="huber"
):
    checkpoints = tuple(checkpoints)
    if not checkpoints or checkpoints != tuple(sorted(set(checkpoints))):
        raise ValueError("Checkpoints must be nonempty and strictly increasing")
    if checkpoints[0] <= 0 or checkpoints[-1] % 2:
        raise ValueError("Checkpoints must be positive with an even final hand count")
    if opponent_name not in OPPONENTS:
        raise ValueError(f"Unknown opponent: {opponent_name}")
    if learn_every <= 0:
        raise ValueError("learn_every must be positive")

    opponent_strategies = {
        "Passive": baseline_strategy(),
        "Random": baseline_strategy(random_play=True),
        "Nash": NashAgent().strategy
    }
    opponents = {
        name: StrategyAgent(strategy, seed=seed + 10_000 + i)
        for i, (name, strategy) in enumerate(opponent_strategies.items())
    }

    # keep the game deals and opponent decisions reproducible across experiments
    rng = random.Random(seed + 20_000)
    learner = DQNAgent(
        learning_rate=learning_rate,
        epsilon=epsilon,
        gamma=1.0,
        batch_size=batch_size,
        replay_capacity=replay_capacity,
        target_update=target_update,
        seed=seed,
        loss=loss
    )
    visits = {}
    results = []
    previous = 0
    training_seconds = 0.0

    for target in checkpoints:
        start = time.perf_counter()
        for hand in range(previous, target):
            # select a fresh opponent each hand for mixed training
            current_opponent = (
                rng.choice(("Passive", "Random", "Nash"))
                if opponent_name == "Mixed" else opponent_name
            )

            # use equal seat exposure just like our tabular q learning baseline
            play_training_hand(
                learner, opponents[current_opponent], hand % 2, rng, visits
            )

            # one gradient update every 16 hands by default keeps training manageable
            # replay stores decisions from all previous hands until the buffer fills
            if (hand + 1) % learn_every == 0:
                learner.learn()

        training_seconds += time.perf_counter() - start
        strategy = greedy_strategy(learner)

        # use exact enumeration rather than noisy simulated test matches
        ev_passive = balanced_exact_ev(strategy, opponent_strategies["Passive"])
        ev_random = balanced_exact_ev(strategy, opponent_strategies["Random"])
        ev_nash = balanced_exact_ev(strategy, opponent_strategies["Nash"])
        # mixed selects each training opponent with equal probability each hand
        ev_mixed = (ev_passive + ev_random + ev_nash) / 3
        with torch.no_grad():
            queen_call, queen_fold = learner.policy_net(encode_info_set("Q|b")).tolist()
        queen_gap = queen_call - queen_fold
        queen_action = "call" if queen_call >= queen_fold else "fold"
        vulnerability = exploitability(strategy)
        coverage = sum(sum(counts.values()) > 0 for counts in visits.values())

        results.append({
            "hands": target,
            "ev_vs_passive": ev_passive,
            "ev_vs_random": ev_random,
            "ev_vs_nash": ev_nash,
            "ev_vs_mixed": ev_mixed,
            "q_b_action": queen_action,
            "q_b_gap": queen_gap,
            "exploitability": vulnerability,
            "visited_info_sets": coverage,
            "learning_updates": learner.learning_steps,
            "replay_size": len(learner.memory),
            "training_seconds": training_seconds
        })

        print(
            f"{target:,} hands | "
            f"vs Passive: {ev_passive:+.4f} | "
            f"vs Random: {ev_random:+.4f} | "
            f"vs Nash: {ev_nash:+.4f} | "
            f"vs Mixed: {ev_mixed:+.4f} | "
            f"Q|b: {queen_action} | "
            f"Exploitability: {vulnerability:.4f} | "
            f"Visited: {coverage}/12 | "
            f"Updates: {learner.learning_steps:,}"
        )
        previous = target

    # preserve the older tabular models and separate dqn models by opponent
    # keep original huber model names for backward compatibility
    # mse policies use separate filenames so earlier runs are never overwritten
    filename = f"dqn_{opponent_name.lower()}" + ("_mse" if loss == "mse" else "")
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    policy_path = model_dir / f"{filename}.json"
    network_path = model_dir / f"{filename}.pt"
    metadata_path = model_dir / f"{filename}_metadata.json"
    csv_path = output_dir / f"{filename}.csv"

    # save both the playable strategy and the trained neural network weights
    save_strategy(strategy, policy_path)
    torch.save(learner.policy_net.state_dict(), network_path)

    metadata = {
        "algorithm": "dqn",
        "game": "kuhn",
        "training_opponent": opponent_name,
        "training_hands": checkpoints[-1],
        "checkpoints": list(checkpoints),
        "seed": seed,
        "network_architecture": [7, 32, 32, 2],
        "learning_rate": learning_rate,
        "loss": loss,
        "epsilon": epsilon,
        "gamma": 1.0,
        "batch_size": batch_size,
        "replay_capacity": replay_capacity,
        "target_update": target_update,
        "learn_every": learn_every,
        "learning_updates": learner.learning_steps,
        "training_seconds": training_seconds,
        "evaluation": results[-1]
    }
    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=4, allow_nan=False)

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    assert load_strategy(policy_path) == strategy
    print(f"\nTraining opponent: {opponent_name}")
    print(f"Loss function: {loss}")
    print(f"Training time: {training_seconds:.3f}s")
    print(f"Policy saved to: {policy_path}")
    print(f"Network saved to: {network_path}")
    print(f"Metadata saved to: {metadata_path}")
    print(f"Results saved to: {csv_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a DQN agent for Kuhn Poker")
    parser.add_argument("--opponent", choices=OPPONENTS, default="Random")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--hands", type=int, default=100_000)
    parser.add_argument("--learn-every", type=int, default=16)
    parser.add_argument("--loss", choices=("huber", "mse"), default="huber")
    args = parser.parse_args()

    if args.hands <= 0 or args.hands % 2:
        parser.error("--hands must be a positive even number")

    checkpoints = tuple(h for h in CHECKPOINTS if h < args.hands) + (args.hands,)
    run_experiment(
        checkpoints=checkpoints,
        opponent_name=args.opponent,
        seed=args.seed,
        learn_every=args.learn_every,
        loss=args.loss
    )
