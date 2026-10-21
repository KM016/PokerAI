import csv
from pathlib import Path

from poker_ai.algorithms.cfr import CFRTrainer
from poker_ai.algorithms.cfr_plus import CFRPlusTrainer
from poker_ai.agents.nash_agent import NashAgent
from poker_ai.agents.strategy_agent import StrategyAgent
from poker_ai.evaluation.matches import play_match, balanced_exact_ev


# creates simple baseline strategies for testing our trained agents
def baseline_strategy(random_play=False):
    strategy = {}

    # define the action probabilities for every possible information set
    for card in ["J", "Q", "K"]:
        for history in ["", "c", "b", "cb"]:
            info = f"{card}|{history}"

            # when no bet has been made, either check or bet
            if history in ["", "c"]:
                strategy[info] = (
                    {"check": 0.5, "bet": 0.5}
                    if random_play else
                    {"check": 1.0, "bet": 0.0}
                )
            else:
                # when facing a bet, either call or fold
                strategy[info] = (
                    {"call": 0.5, "fold": 0.5}
                    if random_play else
                    {"call": 0.0, "fold": 1.0}
                )

    return strategy


# compares CFR+ against different opponents using simulations and exact EV
def run_tournament(n_hands=20_000):
    # create and train vanilla CFR and CFR+ for the same number of iterations
    cfr = CFRTrainer()
    cfr_plus = CFRPlusTrainer(averaging_delay=0)

    print("Training agents...")
    cfr.train(10_000)
    cfr_plus.train(10_000)

    # get the average strategies learned by each algorithm
    cfr_strategy = cfr.average_strategy()
    plus_strategy = cfr_plus.average_strategy()

    # define the different strategies we want CFR+ to play against
    opponents = {
        "Nash": NashAgent().strategy,
        "CFR": cfr_strategy,
        "Passive": baseline_strategy(),
        "Random": baseline_strategy(random_play=True)
    }

    results = []

    for i, (name, opponent_strategy) in enumerate(opponents.items()):
        # create agents with separate rngs for reproducible decisions
        agent_a = StrategyAgent(plus_strategy, seed=100 + i)
        agent_b = StrategyAgent(opponent_strategy, seed=200 + i)

        # simulate hands with both agents playing each position equally
        match = play_match(
            agent_a,
            agent_b,
            n_hands=n_hands,
            seed=300 + i
        )

        # calculate the exact expected payoff without simulating hands
        theoretical_ev = balanced_exact_ev(
            plus_strategy,
            opponent_strategy
        )

        # record the simulated results alongside the theoretical EV
        results.append({
            "opponent": name,
            "hands": n_hands,
            "win_rate": match["win_rate_a"],
            "simulated_ev": match["avg_payoff_a"],
            "exact_ev": theoretical_ev
        })

        print(f"\nCFR+ vs {name}")
        print(f"Win rate: {match['win_rate_a']:.2%}")
        print(f"Simulated EV: {match['avg_payoff_a']:.6f}")
        print(f"Exact EV: {theoretical_ev:.6f}")

    # create the results folder if it does not already exist
    output = Path("results/kuhn/agent_tournament.csv")
    output.parent.mkdir(parents=True, exist_ok=True)

    # save the results so we can compare CFR+ against each opponent
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "opponent",
                "hands",
                "win_rate",
                "simulated_ev",
                "exact_ev"
            ]
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {output}")

    return results


if __name__ == "__main__":
    run_tournament()
