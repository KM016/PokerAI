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
from poker_ai.evaluation.evaluate import exploitability as kuhn_exploitability
from poker_ai.evaluation.matches import balanced_exact_ev as kuhn_balanced_ev
from poker_ai.evaluation.leduc_evaluate import (
    baseline_strategy as leduc_baseline_strategy,
    balanced_exact_ev as leduc_balanced_ev,
    exploitability as leduc_exploitability,
    information_sets as leduc_information_sets
)
from poker_ai.experiments.agent_tournament import baseline_strategy as kuhn_baseline_strategy
from poker_ai.games.kuhn import KuhnPoker
from poker_ai.games.leduc import LeducPoker


CHECKPOINTS = (
    100, 200, 500,
    1_000, 2_000, 5_000,
    10_000, 20_000, 50_000,
    100_000
)

OPPONENTS = ("Passive", "Random", "Nash", "CFR+", "Mixed")
SCHEDULES = ("Constant", "Decay alpha", "Decay epsilon", "Decay both")


# play one training hand and update the learners q values
# the same q learning algorithm works for both kuhn and leduc
def play_training_hand(learner, opponent, learner_seat, rng, visits=None, game_name="kuhn"):

    # create a new game using the selected poker rules
    # kuhn deals 2 cards while leduc deals 3 including the future public card
    if game_name == "kuhn":
        game = KuhnPoker()
        game.cards = rng.sample(["J", "Q", "K"], 2)

    elif game_name == "leduc":
        game = LeducPoker(deal=rng.sample(LeducPoker.DECK, 3))

    else:
        raise ValueError("Unknown poker game")

    # keep playing until somebody folds or the game reaches showdown
    while not game.done:

        # if the opponent acts first, let them make their decision
        if game.current_player != learner_seat:
            action = opponent.choose_action(
                game.get_information_set(),
                game.legal_actions()
            )
            game.step(action)
            continue

        # record what the learner knows and which actions are currently legal
        info_set = game.get_information_set()
        legal_actions = game.legal_actions()

        # epsilon greedy selects between exploring and exploiting q values
        action = learner.choose_action(info_set, legal_actions)

        # count how often the learner visits each information set and action
        # we can use these counts to measure how much of the game was experienced
        if visits is not None:
            if info_set not in visits:
                visits[info_set] = {a: 0 for a in legal_actions}
            visits[info_set][action] += 1

        # apply the learners decision to the game
        game.step(action)

        # the opponent may act before the learner gets another decision
        # keep following the opponent until the game ends or our turn returns
        while not game.done and game.current_player != learner_seat:
            opponent_action = opponent.choose_action(
                game.get_information_set(),
                game.legal_actions()
            )
            game.step(opponent_action)

        if game.done:
            # the hand has ended so update using the actual final payoff
            # this includes all chips won or lost throughout the hand
            learner.update(
                info_set,
                action,
                game.get_payoff()[learner_seat],
                legal_actions
            )

        else:
            # the hand is still ongoing and the learner has another decision
            # update using the estimated value of the next learner state
            # reward is zero because we use the final net payoff at termination
            learner.update(
                info_set,
                action,
                0,
                legal_actions,
                game.get_information_set(),
                game.legal_actions()
            )

    # return the final profit or loss for the learner
    return game.get_payoff()[learner_seat]


# convert a learned q table into a complete strategy for gameplay and evaluation
# exploration is removed so the agent selects actions with the highest q values
def greedy_strategy(learner, game_name="kuhn"):

    # kuhn has 12 information sets which we can construct directly
    if game_name == "kuhn":
        states = []

        for card in ("J", "Q", "K"):
            for history in ("", "c", "b", "cb"):
                info = f"{card}|{history}"

                if history in ("", "c"):
                    actions = ("check", "bet")
                else:
                    actions = ("call", "fold")

                states.append((info, actions))

    elif game_name == "leduc":
        # leduc has more information sets because of the public card
        # and the additional betting round and raise action
        # use our evaluator to get the complete set of legal decisions
        states = [
            (info, actions)
            for info, _, actions in leduc_information_sets()
        ]

    else:
        raise ValueError("Unknown poker game")

    strategy = {}

    # construct action probabilities for every information set
    for info, actions in states:

        # if the learner never visited a state, assume initial q values of zero
        # this does not modify the original q table
        values = learner.q_table.get(
            info,
            {action: 0.0 for action in actions}
        )

        # identify the highest estimated action value at this state
        best_value = max(values[action] for action in actions)

        # several actions can share the same best q value
        best_actions = [
            action for action in actions
            if values[action] == best_value
        ]

        # choose the best actions with equal probability
        # actions with smaller q values receive probability zero
        strategy[info] = {
            action: (
                1.0 / len(best_actions)
                if action in best_actions else 0.0
            )
            for action in actions
        }

    return strategy


# adjust the learning rate and exploration rate during training
# reuse the same schedules that we tested in kuhn poker
def learning_rates(schedule, hand, total_hands):
    if schedule not in SCHEDULES:
        raise ValueError(f"Unknown schedule: {schedule}")

    if total_hands <= 0 or not 0 <= hand < total_hands:
        raise ValueError("Hand must be within the training range")

    # gradually decrease selected parameters from 0.1 to 0.01
    progress = hand / max(1, total_hands - 1)
    decayed = 0.1 / (1 + 9 * progress)

    alpha = decayed if schedule in ("Decay alpha", "Decay both") else 0.1
    epsilon = decayed if schedule in ("Decay epsilon", "Decay both") else 0.1

    return alpha, epsilon


# train q learning against a selected opponent and save the learned policy
# the game parameter lets us reuse the experiment for kuhn and leduc
def run_experiment(
    checkpoints=CHECKPOINTS,
    opponent_name="Random",
    schedule="Decay alpha",
    seed=101,
    model_dir=None,
    output_dir=None,
    game="kuhn"
):
    # check that the game and training settings are valid
    if game not in ("kuhn", "leduc"):
        raise ValueError("Unknown poker game")

    checkpoints = tuple(checkpoints)

    if not checkpoints or any(h <= 0 for h in checkpoints):
        raise ValueError("Checkpoints must be positive and nonempty")

    if tuple(sorted(set(checkpoints))) != checkpoints:
        raise ValueError("Checkpoints must be strictly increasing")

    if checkpoints[-1] % 2 != 0:
        raise ValueError("Total hands must be even to balance player positions")

    if schedule not in SCHEDULES:
        raise ValueError(f"Unknown schedule: {schedule}")

    # save kuhn and leduc results in separate directories
    model_dir = Path(model_dir) if model_dir is not None else Path(f"models/{game}")
    output_dir = Path(output_dir) if output_dir is not None else Path(f"results/{game}")

    if game == "kuhn":
        # retain our original kuhn training opponents
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
        # use our saved leduc cfr+ policy as the strong reference opponent
        # cfr+ approximates equilibrium but is not an exact nash strategy
        opponent_strategies = {
            "Passive": leduc_baseline_strategy("Passive"),
            "Random": leduc_baseline_strategy("Random"),
            "CFR+": load_strategy(model_dir / "cfr_plus.json")
        }

        reference_name = "CFR+"
        evaluate_ev = leduc_balanced_ev
        evaluate_exploitability = leduc_exploitability
        total_info_sets = len(leduc_information_sets())

    # only permit opponents that exist in the selected game
    if opponent_name not in (*opponent_strategies, "Mixed"):
        raise ValueError(f"Unknown opponent for {game}: {opponent_name}")

    # create fixed opponent agents so only our q learning agent is trained
    opponents = {
        name: StrategyAgent(strategy, seed=seed + 10_000 + i)
        for i, (name, strategy) in enumerate(opponent_strategies.items())
    }

    # initialise the q table and an independent rng for card dealing
    learner = QLearningAgent(
        alpha=0.1,
        epsilon=0.1,
        gamma=1.0,
        seed=seed
    )
    rng = random.Random(seed + 20_000)

    # store state visits, checkpoint results and training time
    visits = {}
    results = []
    previous = 0
    training_seconds = 0.0
    total_hands = checkpoints[-1]

    for target in checkpoints:
        # measure the training time separately from evaluation time
        start = time.perf_counter()

        # continue from the previous checkpoint without resetting the q table
        for hand in range(previous, target):

            # update the learning and exploration rates for this hand
            learner.alpha, learner.epsilon = learning_rates(
                schedule, hand, total_hands
            )

            # mixed training selects between the three fixed opponents
            # the opponent changes between hands but not during a hand
            current_opponent = (
                rng.choice(tuple(opponent_strategies))
                if opponent_name == "Mixed"
                else opponent_name
            )

            # alternate the learner between player 0 and player 1
            play_training_hand(
                learner,
                opponents[current_opponent],
                hand % 2,
                rng,
                visits,
                game_name=game
            )

        training_seconds += time.perf_counter() - start

        # convert the q table into a full strategy without exploration
        strategy = greedy_strategy(learner, game_name=game)

        # evaluate exact profit against our three fixed opponents
        ev_passive = evaluate_ev(strategy, opponent_strategies["Passive"])
        ev_random = evaluate_ev(strategy, opponent_strategies["Random"])
        ev_reference = evaluate_ev(strategy, opponent_strategies[reference_name])

        # exploitability measures the advantage of an optimal opponent
        vulnerability = evaluate_exploitability(strategy)

        # count the information sets the learner encountered during training
        coverage = sum(
            sum(counts.values()) > 0
            for counts in visits.values()
        )

        # preserve the old kuhn metric name for existing experiments
        # leduc uses ev_vs_cfr_plus because its reference is only approximate
        reference_key = (
            "ev_vs_nash" if game == "kuhn" else "ev_vs_cfr_plus"
        )

        results.append({
            "hands": target,
            "ev_vs_passive": ev_passive,
            "ev_vs_random": ev_random,
            reference_key: ev_reference,
            "exploitability": vulnerability,
            "visited_info_sets": coverage
        })

        print(
            f"{target:,} hands | "
            f"vs Passive: {ev_passive:+.4f} | "
            f"vs Random: {ev_random:+.4f} | "
            f"vs {reference_name}: {ev_reference:+.4f} | "
            f"Exploitability: {vulnerability:.4f} | "
            f"Visited: {coverage}/{total_info_sets}"
        )

        previous = target

    # construct filenames describing the training opponent and schedule
    filename = (
        f"q_learning_{opponent_name.lower()}_"
        f"{schedule.lower().replace(' ', '_')}"
    )

    policy_path = model_dir / f"{filename}.json"
    values_path = model_dir / f"{filename}_q_values.json"
    metadata_path = model_dir / f"{filename}_metadata.json"
    csv_path = output_dir / f"{filename}.csv"

    # save both the playable strategy and the original q table
    save_strategy(strategy, policy_path)
    save_strategy(learner.q_table, values_path)

    # record settings so the experiment can be reproduced
    metadata = {
        "algorithm": "tabular_q_learning",
        "game": game,
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

    # save the exact evaluation results from each training checkpoint
    output_dir.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(results[0])
        )
        writer.writeheader()
        writer.writerows(results)

    # verify that the saved policy matches what we evaluated
    assert load_strategy(policy_path) == strategy

    print(f"\nGame: {game}")
    print(f"Training opponent: {opponent_name}")
    print(f"Learning schedule: {schedule}")
    print(f"Training time: {training_seconds:.3f}s")
    print(f"Policy saved to: {policy_path}")
    print(f"Q values saved to: {values_path}")
    print(f"Metadata saved to: {metadata_path}")
    print(f"Results saved to: {csv_path}")

    return results


if __name__ == "__main__":
    # allow the game and training configuration to be selected from the terminal
    parser = argparse.ArgumentParser(
        description="Train a poker q learning agent"
    )
    parser.add_argument(
        "--game",
        choices=["kuhn", "leduc"],
        default="kuhn"
    )
    parser.add_argument(
        "--opponent",
        choices=OPPONENTS,
        default="Random"
    )
    parser.add_argument(
        "--schedule",
        choices=SCHEDULES,
        default="Decay alpha"
    )
    parser.add_argument("--seed", type=int, default=101)

    # optionally limit training to a smaller number of hands for testing
    parser.add_argument("--hands", type=int, default=None)

    args = parser.parse_args()

    checkpoints = CHECKPOINTS

    if args.hands is not None:
        checkpoints = (
            tuple(h for h in CHECKPOINTS if h < args.hands)
            + (args.hands,)
        )

    run_experiment(
        game=args.game,
        opponent_name=args.opponent,
        schedule=args.schedule,
        seed=args.seed,
        checkpoints=checkpoints
    )
