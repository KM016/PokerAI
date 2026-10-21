import argparse
import csv
import json
from itertools import product
from pathlib import Path

from poker_ai.agents.nash_agent import NashAgent
from poker_ai.agents.strategy_io import load_strategy
from poker_ai.evaluation.evaluate import exact_ev, exploitability
from poker_ai.evaluation.matches import balanced_exact_ev
from poker_ai.experiments.agent_tournament import baseline_strategy


# compare the saved poker policies rather than training another set of agents
# the model labels identify which training method produced each policy
MODEL_FILES = (
    ("CFR", "cfr.json"),
    ("CFR+", "cfr_plus.json"),
    ("Q-learning Random", "q_learning_random_decay_alpha.json"),
    ("Q-learning Mixed", "q_learning_mixed_decay_alpha.json"),
    ("DQN Random Huber", "dqn_random.json"),
    ("DQN Mixed Huber", "dqn_mixed.json"),
    ("DQN Mixed MSE", "dqn_mixed_mse.json"),
)
FIELDS = (
    "agent", "ev_vs_passive", "ev_vs_random", "ev_vs_nash", "ev_vs_mixed",
    "exploitability", "mixed_action_info_sets", "training_opponent",
    "training_budget", "budget_units", "loss", "training_seconds",
    "source_file"
)


# each player has six information sets and two legal actions per information set
# search over pure actions to find the exact maximum ev against the hand level mixture
def optimal_mixed_ev(opponents):
    decisions = {
        0: [(f"{card}|{history}", ("check", "bet") if history == "" else ("call", "fold"))
            for history in ("", "cb") for card in ("J", "Q", "K")],
        1: [(f"{card}|{history}", ("check", "bet") if history == "c" else ("call", "fold"))
            for history in ("c", "b") for card in ("J", "Q", "K")]
    }
    best_by_seat = []
    for seat, infos in decisions.items():
        best = float("-inf")
        for choices in product((0, 1), repeat=len(infos)):
            policy = {
                info: {action: float(i == chosen) for i, action in enumerate(actions)}
                for (info, actions), chosen in zip(infos, choices)
            }
            value = 0.0
            for opponent in opponents.values():
                if seat == 0:
                    value += exact_ev(policy, opponent)[0] / len(opponents)
                else:
                    value += exact_ev(opponent, policy)[1] / len(opponents)
            best = max(best, value)
        best_by_seat.append(best)
    return sum(best_by_seat) / 2


# prevent broken json policies from silently producing incorrect evaluations
def validate_strategy(strategy):
    for card in ("J", "Q", "K"):
        for history in ("", "c", "b", "cb"):
            info = f"{card}|{history}"
            actions = ("check", "bet") if history in ("", "c") else ("call", "fold")
            if info not in strategy or set(strategy[info]) != set(actions):
                raise ValueError(f"Invalid or missing actions at {info}")
            probabilities = [strategy[info][action] for action in actions]
            if any(not isinstance(p, (int, float)) or not 0 <= p <= 1 for p in probabilities):
                raise ValueError(f"Invalid probabilities at {info}")
            if abs(sum(probabilities) - 1.0) > 1e-6:
                raise ValueError(f"Probabilities do not sum to one at {info}")


# only trust the training metadata that was actually saved with each model
def read_metadata(policy_path):
    path = policy_path.with_name(policy_path.stem + "_metadata.json")
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def load_models(model_dir):
    model_dir = Path(model_dir)
    nash = NashAgent().strategy
    validate_strategy(nash)
    strategies = {"Nash": nash}
    metadata = {"Nash": {}}
    sources = {"Nash": "built in NashAgent"}
    missing = []
    for label, filename in MODEL_FILES:
        path = model_dir / filename
        if not path.exists():
            missing.append(filename)
            continue
        strategy = load_strategy(path)
        validate_strategy(strategy)
        strategies[label] = strategy
        metadata[label] = read_metadata(path)
        sources[label] = str(path)
    return strategies, metadata, sources, missing


# output a csv even if some older policy files are not present
# missing models are explicitly recorded in the terminal and markdown report
def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_metrics(strategies, metadata, sources, opponents):
    rows = []
    for label, strategy in strategies.items():
        ev = {
            name: balanced_exact_ev(strategy, other)
            for name, other in opponents.items()
        }
        info = metadata[label]
        budget = info.get("training_hands", "")
        units = "hands" if budget != "" else ""
        if label in ("CFR", "CFR+"):
            budget = info.get("training_iterations", info.get("iterations", ""))
            units = "iterations" if budget != "" else ""
        if label == "Nash":
            budget, units = "", ""
        rows.append({
            "agent": label,
            "ev_vs_passive": ev["Passive"],
            "ev_vs_random": ev["Random"],
            "ev_vs_nash": ev["Nash"],
            "ev_vs_mixed": sum(ev.values()) / 3,
            "exploitability": exploitability(strategy),
            "mixed_action_info_sets": sum(
                0 < min(actions.values()) < 1 for actions in strategy.values()
            ),
            "training_opponent": info.get("training_opponent", ""),
            "training_budget": budget,
            "budget_units": units,
            "loss": info.get("loss", ""),
            "training_seconds": info.get("training_seconds", ""),
            "source_file": sources[label]
        })
    return rows


def head_to_head(strategies):
    return [
        {"agent": name_a, "opponent": name_b,
         "balanced_exact_ev": balanced_exact_ev(strategy_a, strategy_b)}
        for name_a, strategy_a in strategies.items()
        for name_b, strategy_b in strategies.items()
    ]


def markdown_report(rows, pairings, missing, optimum):
    content = [
        "# Kuhn Poker final benchmark", "",
        "## What we are comparing", "",
        "CFR and CFR+ target low exploitability. Q-learning and DQN were trained to earn rewards against particular opponents. They are solving different problems, so there is no single overall winner.", "",
        "These results use exact game-tree enumeration, not noisy simulated matches. Every agent is evaluated in both seats with equal weight.", "",
        "Mixed means Passive, Random or Nash is selected once per hand with equal probability.", "",
        "## Final strategy performance", "",
        "| Agent | EV vs Passive | EV vs Random | EV vs Nash | EV vs Mixed | Exploitability |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"
    ]
    for row in rows:
        content.append(
            f"| {row['agent']} | {row['ev_vs_passive']:+.4f} | "
            f"{row['ev_vs_random']:+.4f} | {row['ev_vs_nash']:+.4f} | "
            f"{row['ev_vs_mixed']:+.4f} | {row['exploitability']:.6f} |"
        )
    content.extend([
        "", "## Benchmarks and interpretation", "",
        f"The maximum possible balanced EV against the hand-level Mixed distribution is {optimum:+.6f}. This is calculated by enumerating all pure decisions for each player seat separately.",
        "",
        "Lower exploitability is better. Zero means no opponent can systematically gain by best responding to the policy beyond the value of the game.",
        "",
        "A strategy can achieve the highest possible EV against Mixed and still be exploitable. That is why high EV and low exploitability should be reported separately.",
        "",
        "The greedy policies produced by Q-learning and DQN usually choose one action per information set. CFR and Nash can deliberately mix actions, which is essential to equilibrium play.",
        "",
        "## Training context", "",
        "| Agent | Training opponent | Budget | Loss | Recorded training seconds |",
        "| --- | --- | --- | --- | ---: |"
    ])
    for row in rows:
        budget = f"{row['training_budget']} {row['budget_units']}".strip() or "not recorded"
        elapsed = (f"{float(row['training_seconds']):.3f}"
                   if row['training_seconds'] != "" else "not recorded")
        content.append(
            f"| {row['agent']} | {row['training_opponent'] or 'not recorded'} | "
            f"{budget} | {row['loss'] or 'not recorded'} | {elapsed} |"
        )
    content.extend([
        "", "A CFR iteration traverses the game tree, whereas a reinforcement learning hand samples one game. Their training budgets are not directly equivalent.",
        "",
        "Timing values come from model metadata when available. They may have been recorded on different machines or with different code versions, so they are not a controlled speed benchmark.",
        "",
        "## Pairwise evaluation", "",
        "The file kuhn_final_head_to_head.csv contains balanced exact EV for every loaded agent against every other loaded agent. Each value is from the perspective of the agent in that row.",
        "",
        "## Files used", ""
    ])
    for row in rows:
        content.append(f"- {row['agent']}: {row['source_file']}")
    content.extend(["", "## Missing models", ""])
    if missing:
        content.extend(f"- {item}" for item in missing)
    else:
        content.append("All expected saved models were present.")
    content.extend(["", "## Next stage", "",
                    "Kuhn Poker has only 12 information sets. The next useful test is a larger game such as Leduc Poker, where state representation, exploration and scalability matter more.", ""])
    return "\n".join(content)


def run_benchmark(model_dir="models/kuhn", output_dir="results/kuhn"):
    strategies, metadata, sources, missing = load_models(model_dir)
    opponents = {
        "Passive": baseline_strategy(),
        "Random": baseline_strategy(random_play=True),
        "Nash": NashAgent().strategy
    }
    optimum = optimal_mixed_ev(opponents)
    rows = build_metrics(strategies, metadata, sources, opponents)
    pairings = head_to_head(strategies)
    output = Path(output_dir)
    write_csv(output / "kuhn_final_benchmark.csv", rows, FIELDS)
    write_csv(output / "kuhn_final_head_to_head.csv", pairings,
              ("agent", "opponent", "balanced_exact_ev"))
    report_path = output / "kuhn_final_benchmark.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(markdown_report(rows, pairings, missing, optimum), encoding="utf-8")

    print("\n--- Kuhn Poker final benchmark ---")
    for row in rows:
        print(f"{row['agent']:<20} "
              f"Mixed EV {row['ev_vs_mixed']:+.4f} | "
              f"Random EV {row['ev_vs_random']:+.4f} | "
              f"Exploitability {row['exploitability']:.6f}")
    print(f"\nOptimal EV vs Mixed: {optimum:+.6f}")
    if missing:
        print("Missing saved policies: " + ", ".join(missing))
    print(f"Results and report saved to: {output}")
    return rows, pairings, missing


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare saved Kuhn Poker strategies")
    parser.add_argument("--model-dir", default="models/kuhn")
    parser.add_argument("--output-dir", default="results/kuhn")
    args = parser.parse_args()
    run_benchmark(model_dir=args.model_dir, output_dir=args.output_dir)
