import csv
import math
import random
from pathlib import Path
from statistics import mean, stdev

import matplotlib

# use a non interactive backend so plots can be saved without opening windows
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "figure.facecolor": "lightsteelblue",
    "axes.facecolor": "lemonchiffon",
    "savefig.facecolor": "auto",
    "axes.titleweight": "bold",
    "axes.titlepad": 14,
    "grid.alpha": 0.3
})

from poker_ai.agents.strategy_agent import StrategyAgent
from poker_ai.evaluation.leduc_evaluate import balanced_exact_ev
from poker_ai.experiments.compare_leduc_agents import load_agents
from poker_ai.games.leduc import LeducPoker


# default settings for our final monte carlo experiment
SESSIONS = 20
HANDS_PER_SESSION = 1_000
SEED = 101

# 95 percent student t critical value for 19 degrees of freedom
# 20 independent sessions give 20 - 1 = 19 degrees of freedom
T_CRITICAL_95 = 2.093024054


# simulate one independent session between two fixed strategies
# each agent plays half its hands as player 0 and half as player 1
def simulate_session(strategy_a, strategy_b, hands=HANDS_PER_SESSION, seed=SEED):
    # require an even number of hands to balance player positions
    if hands <= 0 or hands % 2 != 0:
        raise ValueError("Hands must be a positive even number")

    # separate seeded rngs control card deals and agent decisions
    rng = random.Random(seed)
    agent_a = StrategyAgent(strategy_a, seed=seed + 1)
    agent_b = StrategyAgent(strategy_b, seed=seed + 2)

    # reuse the game object but reset all game state before each hand
    game = LeducPoker(seed=seed)

    # track the total net profit earned by agent a
    total_payoff = 0

    # each iteration plays two hands using the same card deal
    for _ in range(hands // 2):
        deal = rng.sample(LeducPoker.DECK, 3)

        # first agent a plays as player 0, then the positions are swapped
        for a_first in (True, False):
            game.reset(deal=deal)

            # map the agents onto the two player positions
            agents = (
                (agent_a, agent_b)
                if a_first else (agent_b, agent_a)
            )

            # play until somebody folds or the hand reaches showdown
            while not game.done:
                player = game.current_player

                # the agent only receives its visible information set
                info_set = game.get_information_set()
                legal_actions = game.legal_actions()

                # sample an action from the agents saved strategy
                action = agents[player].choose_action(
                    info_set, legal_actions
                )
                game.step(action)

            # record agent as payoff from the correct player position
            a_seat = 0 if a_first else 1
            total_payoff += game.get_payoff()[a_seat]

    # return average realised profit per hand rather than total profit
    return total_payoff / hands


# summarise the repeated sessions using a confidence interval for mean ev
def summarise_sessions(values, exact_ev):
    # this experiment uses exactly 20 independent sessions
    if len(values) != SESSIONS:
        raise ValueError("Expected exactly 20 sessions")

    # sample mean estimates the expected profit across repeated sessions
    simulated_ev = mean(values)

    # measure the variation between session level average payoffs
    standard_deviation = stdev(values)

    # uncertainty in the estimated mean decreases with more sessions
    standard_error = standard_deviation / math.sqrt(len(values))

    # use a student t confidence interval with 19 degrees of freedom
    margin = T_CRITICAL_95 * standard_error

    return {
        "exact_ev": exact_ev,
        "simulated_ev": simulated_ev,
        "session_sd": standard_deviation,
        "standard_error": standard_error,
        "ci_lower_95": simulated_ev - margin,
        "ci_upper_95": simulated_ev + margin,
        "absolute_error": abs(simulated_ev - exact_ev),
        "exact_ev_inside_ci": (
            simulated_ev - margin <= exact_ev <= simulated_ev + margin
        )
    }


# plot the simulated mean and confidence interval against theoretical ev
def plot_comparison(summaries, output_dir):
    # sort matchups by theoretical payoff to make the chart easier to read
    rows = sorted(summaries, key=lambda row: row["exact_ev"])

    labels = [
        f"{row['agent_a']} vs {row['agent_b']}"
        for row in rows
    ]

    positions = list(range(len(rows)))

    simulated = [row["simulated_ev"] for row in rows]
    exact = [row["exact_ev"] for row in rows]

    # calculate confidence interval distances for horizontal error bars
    errors = [
        row["simulated_ev"] - row["ci_lower_95"]
        for row in rows
    ]

    # make one horizontal comparison plot for all 15 matchups
    fig, ax = plt.subplots(figsize=(12, 9))

    # circles show the monte carlo estimates and bars show uncertainty
    ax.errorbar(
        simulated,
        positions,
        xerr=errors,
        fmt="o",
        capsize=4,
        label="Simulated EV with 95% CI"
    )

    # crosses show the exact expected payoff without sampling error
    ax.scatter(
        exact,
        positions,
        marker="x",
        s=65,
        label="Exact EV",
        zorder=3
    )

    # zero profit separates favourable and unfavourable matchups
    ax.axvline(0, linestyle="--", linewidth=1)

    ax.set_yticks(positions, labels)
    ax.set_xlabel("Expected net profit for first agent (chips per hand)")
    ax.set_title("Leduc Poker: Monte Carlo vs Exact Expected Value")
    ax.grid(axis="x", alpha=0.2)
    ax.legend()
    ax.invert_yaxis()

    fig.tight_layout()

    # save the figure for the readme and future gui
    path = output_dir / "monte_carlo_comparison.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)

    return path


# show the realised payoff of each session for the close cfr vs cfr+ matchup
def plot_session_variance(session_rows, summaries, output_dir):
    # find the cfr vs cfr+ matchup from our saved session results
    chosen = [
        row for row in session_rows
        if row["agent_a"] == "CFR" and row["agent_b"] == "CFR+"
    ]

    # custom subsets used during testing may not contain this matchup
    if not chosen:
        return None

    # find its summary containing the exact payoff and confidence interval
    summary = next(
        row for row in summaries
        if row["agent_a"] == "CFR" and row["agent_b"] == "CFR+"
    )

    sessions = [row["session"] for row in chosen]
    payoffs = [row["simulated_ev"] for row in chosen]

    fig, ax = plt.subplots(figsize=(10, 5.5))

    # plot the realised profit from each independent 1000 hand session
    ax.plot(
        sessions,
        payoffs,
        marker="o",
        label="Realised session EV"
    )

    # show the theoretical expected payoff as a horizontal reference line
    ax.axhline(
        summary["exact_ev"],
        linestyle="--",
        label="Exact EV"
    )

    # shade the confidence interval for the mean across all sessions
    # this is not a prediction interval for individual session outcomes
    ax.axhspan(
        summary["ci_lower_95"],
        summary["ci_upper_95"],
        alpha=0.15,
        label="95% CI for mean EV"
    )

    ax.axhline(0, linestyle=":", linewidth=1)
    ax.set_xticks(sessions)
    ax.set_xlabel("Independent simulation session")
    ax.set_ylabel("Realised profit (chips per hand)")
    ax.set_title("CFR vs CFR+: Finite-Sample Variation")
    ax.grid(alpha=0.2)
    ax.legend()

    fig.tight_layout()

    path = output_dir / "monte_carlo_cfr_variance.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)

    return path


# run monte carlo simulations for every unique matchup
def run_experiment(
    model_dir="models/leduc",
    output_dir="results/leduc",
    model_files=None,
    hands_per_session=HANDS_PER_SESSION
):
    # reject invalid session sizes before any simulation begins
    if hands_per_session <= 0 or hands_per_session % 2:
        raise ValueError("Hands per session must be positive and even")

    # use the existing comparison code to load and validate the saved agents
    agents = load_agents(model_dir, model_files)
    names = list(agents)

    if len(names) < 2:
        raise ValueError("At least two agents are required")

    session_rows = []
    summaries = []

    print("\n--- LEDUC MONTE CARLO VALIDATION ---")
    print(
        f"{SESSIONS} sessions x {hands_per_session:,} hands "
        f"per matchup"
    )

    # compare each unique pair once rather than repeating reversed matchups
    pair_index = 0

    for i, name_a in enumerate(names):
        for name_b in names[i + 1:]:
            strategy_a = agents[name_a]
            strategy_b = agents[name_b]

            # calculate exact ev using the same evaluator as our final benchmark
            theoretical = balanced_exact_ev(
                strategy_a, strategy_b
            )

            values = []

            # run independent sessions with distinct random seeds
            for session in range(1, SESSIONS + 1):
                session_seed = (
                    SEED + pair_index * 100_000 + session * 17
                )

                simulated = simulate_session(
                    strategy_a,
                    strategy_b,
                    hands=hands_per_session,
                    seed=session_seed
                )

                values.append(simulated)

                # save each session separately so we can study variation
                session_rows.append({
                    "agent_a": name_a,
                    "agent_b": name_b,
                    "session": session,
                    "hands": hands_per_session,
                    "simulated_ev": simulated,
                    "exact_ev": theoretical
                })

            # calculate the sample mean, sd and confidence interval
            summary = summarise_sessions(values, theoretical)

            summary.update({
                "agent_a": name_a,
                "agent_b": name_b,
                "sessions": SESSIONS,
                "hands_per_session": hands_per_session
            })

            summaries.append(summary)

            print(
                f"{name_a} vs {name_b} | "
                f"Exact: {theoretical:+.4f} | "
                f"Simulated: {summary['simulated_ev']:+.4f} | "
                f"95% CI: "
                f"[{summary['ci_lower_95']:+.4f}, "
                f"{summary['ci_upper_95']:+.4f}]"
            )

            pair_index += 1

    # create the directory for tables and figures
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # save all 300 session level results
    sessions_path = output_dir / "monte_carlo_sessions.csv"

    with sessions_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(session_rows[0])
        )
        writer.writeheader()
        writer.writerows(session_rows)

    # save one statistical summary for each unique matchup
    summary_path = output_dir / "monte_carlo_summary.csv"

    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(summaries[0])
        )
        writer.writeheader()
        writer.writerows(summaries)

    # generate the overview and detailed variance plots
    overview_path = plot_comparison(summaries, output_dir)
    variance_path = plot_session_variance(
        session_rows, summaries, output_dir
    )

    print(f"\nSession results: {sessions_path}")
    print(f"Statistical summary: {summary_path}")
    print(f"Comparison plot: {overview_path}")

    if variance_path is not None:
        print(f"Session variance plot: {variance_path}")

    return session_rows, summaries


if __name__ == "__main__":
    run_experiment()
