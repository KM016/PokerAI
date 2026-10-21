import csv
import math
from pathlib import Path

from poker_ai.agents.strategy_io import load_strategy
from poker_ai.evaluation.leduc_evaluate import (
    balanced_exact_ev,
    baseline_strategy,
    exploitability,
    information_sets
)


# saved policies from our six trained leduc agents
# these are the final models produced by our training experiments
MODEL_FILES = {
    "CFR": "cfr.json",
    "CFR+": "cfr_plus.json",
    "Q-learning Random": "q_learning_random_decay_alpha.json",
    "Q-learning Mixed": "q_learning_mixed_decay_alpha.json",
    "DQN Random": "dqn_random_mse.json",
    "DQN Mixed": "dqn_mixed_mse.json"
}


# check that a loaded strategy can legally play every leduc information set
def validate_strategy(strategy, name):
    # get every information set and its legal actions from the game evaluator
    expected = {
        info: set(actions)
        for info, _, actions in information_sets()
    }

    # all trained strategies should contain exactly 288 information sets
    if not isinstance(strategy, dict) or set(strategy) != set(expected):
        raise ValueError(
            f"{name} does not contain the correct Leduc information sets"
        )

    # validate the probability distribution at each information set
    for info, legal in expected.items():
        probabilities = strategy[info]

        # reject missing actions or actions that the game does not allow
        if not isinstance(probabilities, dict) or set(probabilities) != legal:
            raise ValueError(f"{name} has incorrect actions at {info}")

        # action probabilities must be finite and nonnegative
        if any(
            not isinstance(probability, (int, float))
            or not math.isfinite(probability)
            or probability < 0
            for probability in probabilities.values()
        ):
            raise ValueError(f"{name} has invalid probabilities at {info}")

        # the probabilities must add up to one
        if abs(sum(probabilities.values()) - 1.0) > 1e-6:
            raise ValueError(f"{name} probabilities do not sum to one at {info}")


# load all trained agents and make sure their saved strategies are valid
def load_agents(model_dir, model_files=None):
    if model_files is None:
        model_files = MODEL_FILES

    model_dir = Path(model_dir)

    # check for missing files before starting the comparison
    missing = [
        str(model_dir / filename)
        for filename in model_files.values()
        if not (model_dir / filename).is_file()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing saved Leduc strategies:\n" + "\n".join(missing)
        )

    agents = {}

    # load each saved json policy into memory
    for name, filename in model_files.items():
        strategy = load_strategy(model_dir / filename)

        # reject invalid policies before running any matches
        validate_strategy(strategy, name)

        agents[name] = strategy

    return agents


# compare all trained leduc strategies using exact expected payoffs
def run_comparison(
    model_dir="models/leduc",
    output_dir="results/leduc",
    model_files=None
):
    # load the strategies without retraining the algorithms
    agents = load_agents(model_dir, model_files)

    if len(agents) < 2:
        raise ValueError("At least two agents are required")

    if "CFR+" not in agents:
        raise ValueError("CFR+ is required as the reference agent")

    # preserve the same agent ordering throughout the exported files
    names = list(agents)

    # create a square matrix containing expected profit for each pairing
    # matrix[a][b] is the expected profit of agent a against agent b
    matrix = {
        name: {opponent: 0.0 for opponent in names}
        for name in names
    }

    print("\n--- LEDUC HEAD TO HEAD COMPARISON ---")

    # compare every unique pair of agents
    # self play is zero after balancing the two player positions
    for i, name_a in enumerate(names):
        for name_b in names[i + 1:]:

            # evaluate every possible deal and action without monte carlo noise
            # both agents are evaluated from each player position
            ev = balanced_exact_ev(
                agents[name_a],
                agents[name_b]
            )

            # leduc is a zero sum game so the reverse payoff is negative
            # we only need one expensive evaluation per unique pairing
            matrix[name_a][name_b] = ev
            matrix[name_b][name_a] = -ev

            print(f"{name_a} vs {name_b}: {ev:+.6f}")

    # load simple baseline opponents for the final comparison summary
    passive = baseline_strategy("Passive")
    random_opponent = baseline_strategy("Random")

    summaries = []

    print("\n--- AGENT PERFORMANCE SUMMARY ---")

    for name in names:
        strategy = agents[name]

        # calculate expected payoff against the simple baselines
        ev_passive = balanced_exact_ev(strategy, passive)
        ev_random = balanced_exact_ev(strategy, random_opponent)

        # reuse the exact cfr+ result from our head to head matrix
        # this avoids performing another evaluation of the same pair
        ev_cfr_plus = matrix[name]["CFR+"]

        # an equal mixture gives each baseline opponent one third probability
        ev_mixed = (ev_passive + ev_random + ev_cfr_plus) / 3

        # average profit against the other trained agents in this tournament
        # this is specific to our chosen set of opponents
        other_agents = [
            opponent for opponent in names
            if opponent != name
        ]

        mean_ev = sum(
            matrix[name][opponent]
            for opponent in other_agents
        ) / len(other_agents)

        # exploitability measures the benefit of an optimal best response
        # lower exploitability indicates greater strategic robustness
        vulnerability = exploitability(strategy)

        # store the final results for future plots and the gui
        summaries.append({
            "agent": name,
            "ev_vs_passive": ev_passive,
            "ev_vs_random": ev_random,
            "ev_vs_cfr_plus": ev_cfr_plus,
            "ev_vs_mixed": ev_mixed,
            "mean_ev_vs_other_agents": mean_ev,
            "exploitability": vulnerability
        })

        print(
            f"{name} | "
            f"vs Random: {ev_random:+.4f} | "
            f"vs CFR+: {ev_cfr_plus:+.4f} | "
            f"Mean vs agents: {mean_ev:+.4f} | "
            f"Exploitability: {vulnerability:.4f}"
        )

    # create the results directory if it does not already exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # save the full head to head matrix
    matrix_path = output_dir / "head_to_head_matrix.csv"

    with matrix_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        # the first row identifies the opponents represented by each column
        writer.writerow(["agent", *names])

        # each following row contains the payoff of that agent against others
        for name in names:
            writer.writerow([
                name,
                *(matrix[name][opponent] for opponent in names)
            ])

    # save a separate summary for plotting and performance comparisons
    summary_path = output_dir / "agent_summary.csv"

    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(summaries[0])
        )
        writer.writeheader()
        writer.writerows(summaries)

    print(f"\nHead to head matrix saved to: {matrix_path}")
    print(f"Agent summary saved to: {summary_path}")

    return matrix, summaries


if __name__ == "__main__":
    run_comparison()
