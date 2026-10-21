import argparse
import csv
import math
from copy import deepcopy
from itertools import product
from pathlib import Path

from poker_ai.agents.nash_agent import NashAgent
from poker_ai.agents.strategy_io import load_strategy
from poker_ai.evaluation.evaluate import exact_ev, exploitability
from poker_ai.evaluation.matches import balanced_exact_ev
from poker_ai.experiments.agent_tournament import baseline_strategy


# compare the saved dqn strategies without retraining either network
SEEDS = (42, 43, 44, 45, 46)
CARDS = ("J", "Q", "K")
HISTORIES = ("", "c", "b", "cb")
INFO_SETS = tuple(f"{card}|{history}" for history in HISTORIES for card in CARDS)
ACTIONS = {
    "": ("check", "bet"),
    "c": ("check", "bet"),
    "b": ("call", "fold"),
    "cb": ("call", "fold")
}


def validate_strategy(strategy):
    # exact evaluation needs every information set and a valid action distribution
    if set(strategy) != set(INFO_SETS):
        raise ValueError("Strategy must contain all 12 Kuhn information sets")

    for info in INFO_SETS:
        actions = ACTIONS[info.split("|")[1]]
        probs = strategy[info]
        if not isinstance(probs, dict) or set(probs) != set(actions):
            raise ValueError(f"Invalid actions at {info}")
        values = [probs[action] for action in actions]
        if any(not isinstance(v, (int, float)) or not math.isfinite(v)
               or v < 0 or v > 1 for v in values):
            raise ValueError(f"Invalid probabilities at {info}")
        if not math.isclose(sum(values), 1.0, abs_tol=1e-8):
            raise ValueError(f"Action probabilities do not sum to 1 at {info}")


def chosen_action(strategy, info):
    # the saved dqn policies are greedy, so each information set has one chosen action
    actions = ACTIONS[info.split("|")[1]]
    probs = strategy[info]
    if not any(math.isclose(probs[action], 1.0, abs_tol=1e-8) for action in actions):
        raise ValueError(f"Expected a deterministic DQN strategy at {info}")
    return max(actions, key=lambda action: probs[action])


def saved_policy_path(model_dir, opponent, seed, learn_every):
    # each seed from the stability experiment has its own folder
    run_name = f"{opponent.lower()}_seed_{seed}_every_{learn_every}"
    return Path(model_dir) / "dqn_stability" / run_name / f"dqn_{opponent.lower()}.json"


def measure_strategy(strategy, opponents, with_exploitability=True):
    # exact ev averages both player seats, rather than estimating from played hands
    scores = {
        name: balanced_exact_ev(strategy, opponent)
        for name, opponent in opponents.items()
    }
    # mixed selects one of these three fixed opponents with equal probability
    scores["Mixed"] = (scores["Passive"] + scores["Random"] + scores["Nash"]) / 3
    if with_exploitability:
        scores["exploitability"] = exploitability(strategy)
    return scores


def optimal_mixed_strategy(opponents):
    # enumerate all 64 deterministic decisions for each seat separately
    # this gives an exact best response to the mixture without training
    # mixing opponents happens once per hand, so evaluate each opponent separately
    strategy = {}
    best_seat_values = []
    for seat, histories in ((0, ("", "cb")), (1, ("c", "b"))):
        infos = [f"{card}|{history}" for history in histories for card in CARDS]
        legal = [ACTIONS[info.split("|")[1]] for info in infos]
        best_value = float("-inf")
        best_actions = None
        for choices in product(*legal):
            candidate = {
                info: {action: float(action == chosen) for action in options}
                for info, options, chosen in zip(infos, legal, choices)
            }
            values = [
                exact_ev(candidate, opponent)[0] if seat == 0
                else exact_ev(opponent, candidate)[1]
                for opponent in opponents.values()
            ]
            score = sum(values) / len(values)
            if score > best_value + 1e-12:
                best_value = score
                best_actions = candidate
        strategy.update(best_actions)
        best_seat_values.append(best_value)
    return strategy, sum(best_seat_values) / 2


def changed_action_strategy(strategy, info, replacement):
    # change only one information set, keeping all other decisions identical
    changed = deepcopy(strategy)
    changed[info] = replacement[info].copy()
    return changed


def alternate_action_strategy(strategy, info):
    # flip one deterministic action to check if the mixed agent can improve locally
    changed = deepcopy(strategy)
    first, second = ACTIONS[info.split("|")[1]]
    current = chosen_action(strategy, info)
    other = second if current == first else first
    changed[info] = {first: float(other == first), second: float(other == second)}
    return changed


def save_csv(path, rows):
    # keep the full results available for followup plots and analysis
    if not rows:
        raise ValueError("Cannot save an empty table")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_report(metrics, actions, differences, opportunities, seeds, optimal_mixed_ev):
    # summarise all strategies and give one action table for each matched seed
    lines = [
        "# DQN strategy analysis",
        "",
        "These are saved greedy policies from the DQN stability experiment.",
        "No models were retrained or modified during this analysis.",
        f"The exact maximum EV against the Mixed opponent distribution is {optimal_mixed_ev:+.6f}.",
        "",
        "## Final exact evaluation",
        "",
        "| Seed | Trained on | EV vs Random | EV vs Mixed | Gap to optimum | Exploitability |",
        "| --- | --- | ---: | ---: | ---: | ---: |"
    ]
    for row in metrics:
        lines.append(
            f"| {row['seed']} | {row['training_opponent']} | "
            f"{row['ev_vs_random']:+.4f} | {row['ev_vs_mixed']:+.4f} | "
            f"{row['gap_to_mixed_optimum']:.4f} | {row['exploitability']:.4f} |"
        )

    lines.extend([
        "",
        "## What the action differences mean",
        "",
        "A disagreement means the two saved agents choose different actions at that information set.",
        "The one state swap columns measure what happens if we replace only the Mixed agent decision",
        "with the Random agent decision while leaving its other 11 decisions unchanged.",
        "A positive change improves EV and a negative change reduces EV.",
        "A zero change can mean the information set is never reached under the current strategy.",
        "These isolated changes are not additive and do not establish the full cause of the EV gap.",
        "The optimal reference is one of potentially several optimal strategies.",
        "A different decision is not automatically an error if it is tied or unreachable.",
        "",
        "## Information set actions by seed",
        ""
    ])
    for seed in seeds:
        lines.extend([
            f"### Seed {seed}",
            "",
            "| Information set | Random trained | Mixed trained | Optimal Mixed reference | Gain using Random action | Gain using optimal action |",
            "| --- | --- | --- | --- | ---: | ---: |"
        ])
        for row in actions:
            if row["seed"] != seed:
                continue
            r = (
                f"{row['delta_ev_vs_mixed_if_switched']:+.4f}"
                if row["different"] else ""
            )
            m = (
                f"{row['delta_ev_vs_mixed_if_oracle_switched']:+.4f}"
                if row["optimal_action"] != row["mixed_action"] else ""
            )
            lines.append(
                f"| `{row['info_set']}` | {row['random_action']} | {row['mixed_action']} | "
                f"{row['optimal_action']} | {r} | {m} |"
            )
        lines.append("")

    lines.extend([
        "## Local improvement opportunities for the Mixed agent",
        "",
        "A positive value means flipping that single Mixed decision improves exact EV against Mixed.",
        "Only positive opportunities are shown here. The CSV contains every tested flip.",
        "",
        "| Seed | Information set | Current action | Alternative | EV gain vs Mixed |",
        "| --- | --- | --- | --- | ---: |"
    ])
    positive = [r for r in opportunities if r["delta_ev_vs_mixed_if_flipped"] > 1e-9]
    for row in sorted(positive, key=lambda r: (-r["delta_ev_vs_mixed_if_flipped"], r["seed"])):
        lines.append(
            f"| {row['seed']} | `{row['info_set']}` | {row['current_action']} | "
            f"{row['alternative_action']} | {row['delta_ev_vs_mixed_if_flipped']:+.4f} |"
        )
    if not positive:
        lines.append("| None | | | | |")

    lines.extend([
        "",
        f"There are {len(differences)} disagreements across {len(seeds)} paired seed comparisons.",
        "",
        "The Nash evaluation is a reference for robustness, not the training target.",
        "The DQN training objective is to maximise payoff against the selected training opponent.",
        ""
    ])
    return "\n".join(lines)


def run_analysis(
    seeds=SEEDS,
    learn_every=16,
    model_dir="models/kuhn",
    output_dir="results/kuhn",
    verbose=True
):
    seeds = tuple(seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be nonempty and unique")
    if learn_every <= 0:
        raise ValueError("learn_every must be positive")

    opponents = {
        "Passive": baseline_strategy(),
        "Random": baseline_strategy(random_play=True),
        "Nash": NashAgent().strategy
    }
    output_dir = Path(output_dir)
    optimal_policy, optimal_mixed_ev = optimal_mixed_strategy(opponents)
    metrics = []
    actions = []
    differences = []
    opportunities = []
    for seed in seeds:
        strategies = {}
        for training_opponent in ("Random", "Mixed"):
            path = saved_policy_path(model_dir, training_opponent, seed, learn_every)
            if not path.is_file():
                raise FileNotFoundError(
                    f"Missing saved strategy: {path}\n"
                    "Run python -m poker_ai.experiments.dqn_stability first."
                )
            strategy = load_strategy(path)
            validate_strategy(strategy)
            # the action comparisons rely on the greedy strategies saved by dqn
            for info in INFO_SETS:
                chosen_action(strategy, info)
            strategies[training_opponent] = strategy
            scores = measure_strategy(strategy, opponents)
            metrics.append({
                "seed": seed,
                "training_opponent": training_opponent,
                "ev_vs_passive": scores["Passive"],
                "ev_vs_random": scores["Random"],
                "ev_vs_nash": scores["Nash"],
                "ev_vs_mixed": scores["Mixed"],
                "gap_to_mixed_optimum": optimal_mixed_ev - scores["Mixed"],
                "exploitability": scores["exploitability"],
                "policy_file": str(path)
            })

        random_policy = strategies["Random"]
        mixed_policy = strategies["Mixed"]
        mixed_scores = measure_strategy(mixed_policy, opponents, with_exploitability=False)
        difference_count = 0

        for info in INFO_SETS:
            random_action = chosen_action(random_policy, info)
            mixed_action = chosen_action(mixed_policy, info)
            optimal_action = chosen_action(optimal_policy, info)
            different = random_action != mixed_action
            row = {
                "seed": seed,
                "info_set": info,
                "player_seat": 0 if info.split("|")[1] in ("", "cb") else 1,
                "random_action": random_action,
                "mixed_action": mixed_action,
                "optimal_action": optimal_action,
                "different": different,
                "delta_ev_vs_random_if_switched": 0.0,
                "delta_ev_vs_mixed_if_switched": 0.0,
                "delta_ev_vs_mixed_if_oracle_switched": 0.0
            }
            if different:
                difference_count += 1
                swapped = changed_action_strategy(mixed_policy, info, random_policy)
                changed_scores = measure_strategy(swapped, opponents, with_exploitability=False)
                row["delta_ev_vs_random_if_switched"] = (
                    changed_scores["Random"] - mixed_scores["Random"]
                )
                row["delta_ev_vs_mixed_if_switched"] = (
                    changed_scores["Mixed"] - mixed_scores["Mixed"]
                )
            if optimal_action != mixed_action:
                oracle_swapped = changed_action_strategy(mixed_policy, info, optimal_policy)
                oracle_scores = measure_strategy(
                    oracle_swapped, opponents, with_exploitability=False
                )
                row["delta_ev_vs_mixed_if_oracle_switched"] = (
                    oracle_scores["Mixed"] - mixed_scores["Mixed"]
                )
            if different:
                differences.append(row.copy())
            actions.append(row)

            # test both actions locally, even when the two models already agree
            flipped = alternate_action_strategy(mixed_policy, info)
            flipped_scores = measure_strategy(flipped, opponents, with_exploitability=False)
            opportunities.append({
                "seed": seed,
                "info_set": info,
                "player_seat": row["player_seat"],
                "current_action": mixed_action,
                "alternative_action": chosen_action(flipped, info),
                "delta_ev_vs_random_if_flipped": flipped_scores["Random"] - mixed_scores["Random"],
                "delta_ev_vs_mixed_if_flipped": flipped_scores["Mixed"] - mixed_scores["Mixed"]
            })

        if verbose:
            random_ev = metrics[-2]["ev_vs_mixed"]
            mixed_ev = metrics[-1]["ev_vs_mixed"]
            print(
                f"Seed {seed} | Disagreements: {difference_count}/12 | "
                f"EV vs Mixed, Random trained: {random_ev:+.4f} | "
                f"Mixed trained: {mixed_ev:+.4f} | "
                f"Gap to optimum: {optimal_mixed_ev - mixed_ev:.4f}"
            )
            improving = [
                row for row in opportunities
                if row["seed"] == seed and row["delta_ev_vs_mixed_if_flipped"] > 1e-9
            ]
            if improving:
                best = max(improving, key=lambda r: r["delta_ev_vs_mixed_if_flipped"])
                print(
                    f"  Best one-state change: {best['info_set']} "
                    f"{best['current_action']} -> {best['alternative_action']} | "
                    f"Mixed EV gain: {best['delta_ev_vs_mixed_if_flipped']:+.4f}"
                )
            else:
                print("  No improving single-action flip found")

    save_csv(output_dir / "dqn_strategy_metrics.csv", metrics)
    save_csv(output_dir / "dqn_strategy_actions.csv", actions)
    # a pair can agree in every state, so a disagreement table may be empty
    if differences:
        save_csv(output_dir / "dqn_strategy_differences.csv", differences)
    else:
        path = output_dir / "dqn_strategy_differences.csv"
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(actions[0]))
            writer.writeheader()
    save_csv(output_dir / "dqn_strategy_opportunities.csv", opportunities)
    report = make_report(
        metrics, actions, differences, opportunities, seeds, optimal_mixed_ev
    )
    (output_dir / "dqn_strategy_report.md").write_text(report, encoding="utf-8")
    if verbose:
        print(f"\nOptimal EV vs Mixed: {optimal_mixed_ev:+.6f}")
        print(f"Strategy analysis saved to: {output_dir}")
        print("Open dqn_strategy_report.md to inspect all 12 decisions for every seed.")
    return metrics, actions, differences, opportunities


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyse saved DQN Kuhn strategies")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    parser.add_argument("--learn-every", type=int, default=16)
    parser.add_argument("--model-dir", default="models/kuhn")
    parser.add_argument("--output-dir", default="results/kuhn")
    args = parser.parse_args()
    run_analysis(
        seeds=args.seeds, learn_every=args.learn_every,
        model_dir=args.model_dir, output_dir=args.output_dir
    )
