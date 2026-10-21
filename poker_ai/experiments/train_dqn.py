
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
from poker_ai.algorithms.dqn import (
    CARDS, HISTORIES, LEDUC_ACTIONS, LEDUC_INPUT_SIZE, encode_info_set
)
from poker_ai.evaluation.evaluate import exploitability as kuhn_exploitability
from poker_ai.evaluation.matches import balanced_exact_ev as kuhn_balanced_ev
from poker_ai.evaluation.leduc_evaluate import (
    baseline_strategy as leduc_baseline_strategy,
    balanced_exact_ev as leduc_balanced_ev,
    exploitability as leduc_exploitability,
    information_sets as leduc_information_sets
)
from poker_ai.experiments.agent_tournament import baseline_strategy as kuhn_baseline_strategy
from poker_ai.experiments.train_q_learning import CHECKPOINTS
from poker_ai.games.kuhn import KuhnPoker
from poker_ai.games.leduc import LeducPoker


OPPONENTS = ("Passive", "Random", "Nash", "CFR+", "Mixed")


# play a training hand and store every decision made by the learner
# supports both kuhn and leduc without changing the reinforcement learning rule
def play_training_hand(
    learner, opponent, learner_seat, rng, visits=None, game_name="kuhn"
):
    # make sure the learner is assigned to a valid player position
    if learner_seat not in (0, 1):
        raise ValueError("Learner seat must be 0 or 1")

    # create a new game using the selected rules
    # kuhn deals 2 different cards while leduc deals 3 physical cards
    if game_name == "kuhn":
        game = KuhnPoker()
        game.cards = rng.sample(list(CARDS), 2)

    elif game_name == "leduc":
        game = LeducPoker(deal=rng.sample(LeducPoker.DECK, 3))

    else:
        raise ValueError("Unknown poker game")

    # continue playing until somebody folds or a showdown occurs
    while not game.done:

        # allow the opponent to act first when it is not the learners turn
        if game.current_player != learner_seat:
            action = opponent.choose_action(
                game.get_information_set(),
                game.legal_actions()
            )
            game.step(action)
            continue

        # record the information available to the learner
        info_set = game.get_information_set()
        legal_actions = game.legal_actions()

        # choose an action using epsilon greedy exploration
        action = learner.choose_action(info_set, legal_actions)

        # count how often the learner encounters each information set
        if visits is not None:
            if info_set not in visits:
                visits[info_set] = {
                    choice: 0 for choice in legal_actions
                }
            visits[info_set][action] += 1

        # apply the decision made by the learner
        game.step(action)

        # allow the opponent to respond before the learner acts again
        # the next state must be another learner decision or the end of the hand
        while not game.done and game.current_player != learner_seat:
            opponent_action = opponent.choose_action(
                game.get_information_set(),
                game.legal_actions()
            )
            game.step(opponent_action)

        if game.done:
            # the hand is finished so remember the actual profit or loss
            learner.remember(
                info_set,
                action,
                game.get_payoff()[learner_seat],
                legal_actions
            )

        else:
            # the hand continues so remember the next learner decision
            # the immediate reward is zero because we use final net payoff
            learner.remember(
                info_set,
                action,
                0.0,
                legal_actions,
                game.get_information_set(),
                game.legal_actions()
            )

    # return the final profit or loss earned by the learner
    return game.get_payoff()[learner_seat]


# convert the trained neural network into a complete playable strategy
# this disables exploration and always chooses the best legal predicted action
def greedy_strategy(learner, game_name="kuhn"):
    strategy = {}

    if game_name == "kuhn":
        # construct all 12 possible kuhn information sets
        states = []

        for card in CARDS:
            for history in HISTORIES:
                info = f"{card}|{history}"

                actions = (
                    ("check", "bet")
                    if history in ("", "c")
                    else ("call", "fold")
                )

                states.append((info, actions))

    elif game_name == "leduc":
        # get all 288 leduc information sets and their legal actions
        # we already calculate these in the leduc evaluation framework
        states = [
            (info, actions)
            for info, _, actions in leduc_information_sets()
        ]

    else:
        raise ValueError("Unknown poker game")

    # predictions do not require gradient calculations
    with torch.no_grad():
        for info, legal in states:
            # predict the q values for the current information set
            values = learner.policy_net(learner._encode(info))

            # identify the highest predicted q value among legal actions
            # this is important because leduc has five possible network outputs
            best_action = max(
                legal,
                key=lambda action: values[
                    learner._action_index(action, legal)
                ].item()
            )

            # convert the selected action into a deterministic policy
            # the best action gets probability 1 and other actions get 0
            strategy[info] = {
                action: float(action == best_action)
                for action in legal
            }

    return strategy


# train dqn against fixed opponents and evaluate at different checkpoints
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
    model_dir=None,
    output_dir=None,
    loss="huber",
    game="kuhn"
):
    # validate the requested training configuration
    checkpoints = tuple(checkpoints)

    if not checkpoints or checkpoints != tuple(sorted(set(checkpoints))):
        raise ValueError("Checkpoints must be nonempty and strictly increasing")

    if checkpoints[0] <= 0 or checkpoints[-1] % 2:
        raise ValueError("Checkpoints must be positive with an even final hand count")

    if learn_every <= 0:
        raise ValueError("learn_every must be positive")

    if game not in ("kuhn", "leduc"):
        raise ValueError("Unknown poker game")

    if loss not in ("huber", "mse"):
        raise ValueError("loss must be huber or mse")

    # select the output directories for the chosen game
    # this prevents leduc models from overwriting kuhn models
    model_dir = Path(model_dir) if model_dir is not None else Path(f"models/{game}")
    output_dir = Path(output_dir) if output_dir is not None else Path(f"results/{game}")

    if game == "kuhn":
        # keep the existing kuhn opponents and known nash strategy
        opponent_strategies = {
            "Passive": kuhn_baseline_strategy(),
            "Random": kuhn_baseline_strategy(random_play=True),
            "Nash": NashAgent().strategy
        }

        reference_name = "Nash"
        evaluate_ev = kuhn_balanced_ev
        evaluate_exploitability = kuhn_exploitability
        total_info_sets = 12

    else:
        # use the trained leduc cfr+ strategy as the strong reference opponent
        # this is an approximate equilibrium rather than an exact nash strategy
        opponent_strategies = {
            "Passive": leduc_baseline_strategy("Passive"),
            "Random": leduc_baseline_strategy("Random"),
            "CFR+": load_strategy(model_dir / "cfr_plus.json")
        }

        reference_name = "CFR+"
        evaluate_ev = leduc_balanced_ev
        evaluate_exploitability = leduc_exploitability
        total_info_sets = len(leduc_information_sets())

    # check that the selected opponent exists for this game
    if opponent_name not in (*opponent_strategies, "Mixed"):
        raise ValueError(f"Unknown opponent for {game}: {opponent_name}")

    # convert fixed strategy tables into opponents that can play hands
    opponents = {
        name: StrategyAgent(strategy, seed=seed + 10_000 + i)
        for i, (name, strategy) in enumerate(opponent_strategies.items())
    }

    # initialise a new neural network with the correct game architecture
    learner = DQNAgent(
        learning_rate=learning_rate,
        epsilon=epsilon,
        gamma=1.0,
        batch_size=batch_size,
        replay_capacity=replay_capacity,
        target_update=target_update,
        seed=seed,
        loss=loss,
        game=game
    )

    # use a separate rng to reproduce the same card dealing sequence
    rng = random.Random(seed + 20_000)

    # store visited information sets and experiment results
    visits = {}
    results = []
    previous = 0
    training_seconds = 0.0

    for target in checkpoints:
        # measure training time separately from evaluation time
        start = time.perf_counter()

        # continue training from the previous checkpoint
        for hand in range(previous, target):
            # mixed training selects a new fixed opponent before each hand
            current_opponent = (
                rng.choice(tuple(opponent_strategies))
                if opponent_name == "Mixed"
                else opponent_name
            )

            # alternate player positions to reduce seat imbalance
            play_training_hand(
                learner,
                opponents[current_opponent],
                hand % 2,
                rng,
                visits,
                game_name=game
            )

            # train using a batch from replay memory every 16 hands by default
            if (hand + 1) % learn_every == 0:
                learner.learn()

        training_seconds += time.perf_counter() - start

        # extract a policy without random exploration
        strategy = greedy_strategy(learner, game_name=game)

        # evaluate exact profit against each fixed opponent
        ev_passive = evaluate_ev(
            strategy, opponent_strategies["Passive"]
        )
        ev_random = evaluate_ev(
            strategy, opponent_strategies["Random"]
        )
        ev_reference = evaluate_ev(
            strategy, opponent_strategies[reference_name]
        )

        # mixed assumes equal probability of meeting each opponent
        ev_mixed = (ev_passive + ev_random + ev_reference) / 3

        # calculate exploitability and information set coverage
        vulnerability = evaluate_exploitability(strategy)
        coverage = len(visits)

        # store results common to both kuhn and leduc
        row = {
            "hands": target,
            "ev_vs_passive": ev_passive,
            "ev_vs_random": ev_random
        }

        if game == "kuhn":
            # preserve the queen call diagnostic from our kuhn research
            with torch.no_grad():
                queen_call, queen_fold = learner.policy_net(
                    encode_info_set("Q|b")
                ).tolist()

            row["ev_vs_nash"] = ev_reference
            row["ev_vs_mixed"] = ev_mixed
            row["q_b_action"] = (
                "call" if queen_call >= queen_fold else "fold"
            )
            row["q_b_gap"] = queen_call - queen_fold

        else:
            # leduc uses cfr+ instead of a predefined nash equilibrium
            row["ev_vs_cfr_plus"] = ev_reference
            row["ev_vs_mixed"] = ev_mixed

        # record learning progress and how much experience was collected
        row.update({
            "exploitability": vulnerability,
            "visited_info_sets": coverage,
            "learning_updates": learner.learning_steps,
            "replay_size": len(learner.memory),
            "training_seconds": training_seconds
        })

        results.append(row)

        print(
            f"{target:,} hands | "
            f"vs Passive: {ev_passive:+.4f} | "
            f"vs Random: {ev_random:+.4f} | "
            f"vs {reference_name}: {ev_reference:+.4f} | "
            f"vs Mixed: {ev_mixed:+.4f} | "
            f"Exploitability: {vulnerability:.4f} | "
            f"Visited: {coverage}/{total_info_sets} | "
            f"Updates: {learner.learning_steps:,}"
        )

        previous = target

    # build filenames that distinguish opponents and loss functions
    # retain the original naming convention for kuhn compatibility
    filename = (
        f"dqn_{opponent_name.lower()}"
        + ("_mse" if loss == "mse" else "")
    )

    # create the model and results folders if necessary
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    policy_path = model_dir / f"{filename}.json"
    network_path = model_dir / f"{filename}.pt"
    metadata_path = model_dir / f"{filename}_metadata.json"
    csv_path = output_dir / f"{filename}.csv"

    # save the playable strategy and the trained neural network weights
    save_strategy(strategy, policy_path)
    torch.save(learner.policy_net.state_dict(), network_path)

    # record the architecture corresponding to the selected game
    architecture = (
        [7, 32, 32, 2]
        if game == "kuhn"
        else [LEDUC_INPUT_SIZE, 32, 32, len(LEDUC_ACTIONS)]
    )

    # store the experiment settings so results can be reproduced
    metadata = {
        "algorithm": "dqn",
        "game": game,
        "training_opponent": opponent_name,
        "training_hands": checkpoints[-1],
        "checkpoints": list(checkpoints),
        "seed": seed,
        "network_architecture": architecture,
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

    # save every checkpoint measurement to a csv file
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0])
        )
        writer.writeheader()
        writer.writerows(results)

    # confirm the saved policy matches the strategy we evaluated
    assert load_strategy(policy_path) == strategy

    print(f"\nGame: {game}")
    print(f"Training opponent: {opponent_name}")
    print(f"Loss function: {loss}")
    print(f"Training time: {training_seconds:.3f}s")
    print(f"Policy saved to: {policy_path}")
    print(f"Network saved to: {network_path}")
    print(f"Metadata saved to: {metadata_path}")
    print(f"Results saved to: {csv_path}")

    return results


if __name__ == "__main__":
    # allow the user to choose the game and training settings from the terminal
    parser = argparse.ArgumentParser(
        description="Train a DQN agent for Kuhn or Leduc Poker"
    )

    parser.add_argument(
        "--game",
        choices=("kuhn", "leduc"),
        default="kuhn"
    )
    parser.add_argument(
        "--opponent",
        choices=OPPONENTS,
        default="Random"
    )
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--hands", type=int, default=100_000)
    parser.add_argument("--learn-every", type=int, default=16)
    parser.add_argument(
        "--loss",
        choices=("huber", "mse"),
        default=None
    )

    args = parser.parse_args()

    # each training run must contain a positive even number of hands
    if args.hands <= 0 or args.hands % 2:
        parser.error("--hands must be a positive even number")

    # keep huber as the default for kuhn and use mse for leduc
    loss = args.loss or ("mse" if args.game == "leduc" else "huber")

    # include only checkpoints that fit inside the selected training budget
    checkpoints = (
        tuple(h for h in CHECKPOINTS if h < args.hands)
        + (args.hands,)
    )

    run_experiment(
        checkpoints=checkpoints,
        opponent_name=args.opponent,
        seed=args.seed,
        learn_every=args.learn_every,
        loss=loss,
        game=args.game
    )
