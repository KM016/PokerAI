import argparse
import csv
import json
import random
import time
from copy import deepcopy
from pathlib import Path
from statistics import mean, stdev

import torch
from torch import nn

from poker_ai.agents.dqn_agent import DQNAgent
from poker_ai.agents.nash_agent import NashAgent
from poker_ai.agents.strategy_agent import StrategyAgent
from poker_ai.agents.strategy_io import save_strategy
from poker_ai.algorithms.dqn import encode_info_set
from poker_ai.evaluation.evaluate import exploitability
from poker_ai.evaluation.matches import balanced_exact_ev
from poker_ai.experiments.agent_tournament import baseline_strategy
from poker_ai.experiments.train_dqn import greedy_strategy, play_training_hand
from poker_ai.experiments.train_q_learning import CHECKPOINTS


# isolate exploration and network learning rate while keeping other settings fixed
CONFIGS = {
    "Baseline": {"learning_rate": 0.001, "epsilon": 0.1},
    "More exploration": {"learning_rate": 0.001, "epsilon": 0.3},
    "Lower learning rate": {"learning_rate": 0.0003, "epsilon": 0.1},
    "Squared error loss": {"learning_rate": 0.001, "epsilon": 0.1, "loss": "mse"}
}
SEEDS = (42, 43, 44, 45, 46)
METRICS = (
    "ev_vs_mixed", "ev_vs_random", "exploitability", "q_call_minus_fold",
    "q_call_rate", "q_call_chosen", "q_fold_chosen", "q_visits",
    "q_call_gain", "q_call_locally_optimal"
)


def slug(name):
    return name.lower().replace(" ", "_")


def save_csv(path, rows):
    if not rows:
        raise ValueError("Cannot save an empty table")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def inspect_queen_bet(learner, visits):
    # q|b is holding a queen when facing the opening bet
    # output index 0 is call and index 1 is fold at this information set
    with torch.no_grad():
        call, fold = learner.policy_net(encode_info_set("Q|b")).tolist()

    actions = visits.get("Q|b", {})
    calls = actions.get("call", 0)
    folds = actions.get("fold", 0)
    total = calls + folds
    return {
        "q_call": call,
        "q_fold": fold,
        "q_call_minus_fold": call - fold,
        "q_greedy_action": "call" if call >= fold else "fold",
        "q_call_chosen": calls,
        "q_fold_chosen": folds,
        "q_visits": total,
        # an unvisited state has no defined empirical call frequency
        "q_call_rate": calls / total if total else None
    }


def queen_call_counterfactual(strategy, opponents, original_ev=None):
    # measure the exact change if we force call at q|b without changing anything else
    if original_ev is None:
        original_ev = sum(balanced_exact_ev(strategy, s) for s in opponents.values()) / 3

    modified = deepcopy(strategy)
    modified["Q|b"] = {"call": 1.0, "fold": 0.0}
    new_ev = sum(balanced_exact_ev(modified, s) for s in opponents.values()) / 3
    return new_ev - original_ev


def summarise_runs(runs, checkpoints, configs):
    summary = []
    for configuration in configs:
        for hands in checkpoints:
            selected = [r for r in runs if r["configuration"] == configuration and r["hands"] == hands]
            row = {
                "configuration": configuration,
                "hands": hands,
                "n_seeds": len(selected)
            }
            for metric in METRICS:
                samples = [float(r[metric]) for r in selected if r[metric] is not None]
                row[f"{metric}_mean"] = mean(samples) if samples else None
                row[f"{metric}_sd"] = stdev(samples) if len(samples) > 1 else 0.0 if samples else None
            row["q_call_seeds"] = sum(r["q_greedy_action"] == "call" for r in selected)
            summary.append(row)
    return summary


def run_diagnostic(
    checkpoints=CHECKPOINTS,
    seeds=SEEDS,
    configs=None,
    learn_every=16,
    output_dir="results/kuhn",
    model_dir="models/kuhn",
    verbose=True
):
    # do not change the original training script or overwrite existing dqn models
    checkpoints = tuple(checkpoints)
    seeds = tuple(seeds)
    configs = CONFIGS if configs is None else configs

    if not checkpoints or checkpoints != tuple(sorted(set(checkpoints))):
        raise ValueError("Checkpoints must increase and be unique")
    if checkpoints[0] <= 0 or checkpoints[-1] % 2:
        raise ValueError("Checkpoints must be positive and end at an even hand count")
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Seeds must be nonempty and unique")
    if not configs or len({slug(name) for name in configs}) != len(configs):
        raise ValueError("Configurations must have unique names")
    if learn_every <= 0:
        raise ValueError("learn_every must be positive")
    for params in configs.values():
        if not {"learning_rate", "epsilon"} <= set(params) or set(params) - {"learning_rate", "epsilon", "loss"}:
            raise ValueError("Configurations need learning_rate, epsilon and optional loss")
        if params.get("loss", "huber") not in ("huber", "mse"):
            raise ValueError("Unknown training loss")
        if params["learning_rate"] <= 0 or not 0 <= params["epsilon"] <= 1:
            raise ValueError("Invalid learning rate or exploration rate")

    output_dir = Path(output_dir)
    model_dir = Path(model_dir)
    base_strategies = {
        "Passive": baseline_strategy(),
        "Random": baseline_strategy(random_play=True),
        "Nash": NashAgent().strategy
    }
    results = []

    for configuration, params in configs.items():
        if verbose:
            print(f"\n--- {configuration}: lr={params['learning_rate']}, epsilon={params['epsilon']}, loss={params.get('loss', 'huber')} ---")
        for seed in seeds:
            rng = random.Random(seed + 20_000)
            opponents = {
                name: StrategyAgent(strategy, seed=seed + 10_000 + i)
                for i, (name, strategy) in enumerate(base_strategies.items())
            }
            learner = DQNAgent(
                learning_rate=params["learning_rate"],
                epsilon=params["epsilon"],
                gamma=1.0,
                batch_size=32,
                replay_capacity=10_000,
                target_update=100,
                seed=seed
            )
            # squared error estimates the conditional mean reward rather than a huber compromise
            # this matters for skewed call outcomes at q|b
            if params.get("loss", "huber") == "mse":
                learner.loss_function = nn.MSELoss()
            visits = {}
            previous = 0
            training_seconds = 0.0
            last_strategy = None

            for target in checkpoints:
                start = time.perf_counter()
                for hand in range(previous, target):
                    opponent = rng.choice(("Passive", "Random", "Nash"))
                    play_training_hand(learner, opponents[opponent], hand % 2, rng, visits)
                    if (hand + 1) % learn_every == 0:
                        learner.learn()
                training_seconds += time.perf_counter() - start

                strategy = greedy_strategy(learner)
                ev_passive = balanced_exact_ev(strategy, base_strategies["Passive"])
                ev_random = balanced_exact_ev(strategy, base_strategies["Random"])
                ev_nash = balanced_exact_ev(strategy, base_strategies["Nash"])
                ev_mixed = (ev_passive + ev_random + ev_nash) / 3
                details = inspect_queen_bet(learner, visits)
                gain = queen_call_counterfactual(strategy, base_strategies, ev_mixed)

                results.append({
                    "configuration": configuration,
                    "seed": seed,
                    "hands": target,
                    "learning_rate": params["learning_rate"],
                    "epsilon": params["epsilon"],
                    "loss": params.get("loss", "huber"),
                    "ev_vs_passive": ev_passive,
                    "ev_vs_random": ev_random,
                    "ev_vs_nash": ev_nash,
                    "ev_vs_mixed": ev_mixed,
                    "exploitability": exploitability(strategy),
                    "visited_info_sets": len(visits),
                    "learning_updates": learner.learning_steps,
                    "training_seconds": training_seconds,
                    **details,
                    "q_call_gain": gain,
                    # this flag checks whether the current action is optimal holding all other choices fixed
                    "q_call_locally_optimal": int(details["q_greedy_action"] == "call" or gain <= 1e-10)
                })
                last_strategy = strategy
                previous = target

            # one model per seed and configuration preserves all earlier experiments
            dest = model_dir / "dqn_qb_diagnostic" / f"{slug(configuration)}_seed_{seed}"
            dest.mkdir(parents=True, exist_ok=True)
            save_strategy(last_strategy, dest / "policy.json")
            torch.save(learner.policy_net.state_dict(), dest / "network.pt")
            with (dest / "metadata.json").open("w", encoding="utf-8") as file:
                json.dump({
                    "configuration": configuration,
                    "seed": seed,
                    "hands": checkpoints[-1],
                    "learn_every": learn_every,
                    **params,
                    "final_metrics": results[-1]
                }, file, indent=4, allow_nan=False)
            if verbose:
                last = results[-1]
                print(
                    f"Seed {seed} | Q|b: {last['q_greedy_action']} | "
                    f"Q(call)-Q(fold): {last['q_call_minus_fold']:+.4f} | "
                    f"call/fold samples: {last['q_call_chosen']}/{last['q_fold_chosen']} | "
                    f"EV vs Mixed: {last['ev_vs_mixed']:+.4f} | "
                    f"call gain: {last['q_call_gain']:+.4f} | "
                    f"time: {last['training_seconds']:.1f}s"
                )

    summary = summarise_runs(results, checkpoints, configs)
    save_csv(output_dir / "dqn_qb_diagnostic_runs.csv", results)
    save_csv(output_dir / "dqn_qb_diagnostic_summary.csv", summary)
    if verbose:
        print("\n--- Final mean results ---")
        for row in summary:
            if row["hands"] == checkpoints[-1]:
                print(
                    f"{row['configuration']}: "
                    f"Q|b call in {row['q_call_seeds']}/{row['n_seeds']} seeds | "
                    f"EV vs Mixed {row['ev_vs_mixed_mean']:+.4f} | "
                    f"Q gap {row['q_call_minus_fold_mean']:+.4f} | "
                    f"Exploitability {row['exploitability_mean']:.4f}"
                )
        print(f"\nResults saved to: {output_dir}")
        print(f"Independent models saved under: {model_dir / 'dqn_qb_diagnostic'}")
    return results, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose Q|b learning across DQN configurations")
    parser.add_argument("--hands", type=int, default=100_000)
    parser.add_argument("--learn-every", type=int, default=16)
    args = parser.parse_args()
    if args.hands <= 0 or args.hands % 2:
        parser.error("--hands must be a positive even number")
    checkpoints = tuple(h for h in CHECKPOINTS if h < args.hands) + (args.hands,)
    run_diagnostic(checkpoints=checkpoints, learn_every=args.learn_every)
