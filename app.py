import csv
from html import escape
from itertools import accumulate
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

from poker_ai.agents.nash_agent import NashAgent
from poker_ai.agents.strategy_agent import StrategyAgent
from poker_ai.agents.strategy_io import load_strategy
from poker_ai.evaluation.leduc_evaluate import (
    baseline_strategy as leduc_baseline,
    information_sets as leduc_information_sets
)
from poker_ai.experiments.agent_tournament import (
    baseline_strategy as kuhn_baseline
)
from poker_ai.games.kuhn import KuhnPoker
from poker_ai.games.leduc import LeducPoker


# all models and results are loaded relative to the project directory
ROOT = Path(__file__).resolve().parent

# keep the established background colours for all matplotlib graphs
plt.rcParams.update({
    "figure.facecolor": "lightsteelblue",
    "axes.facecolor": "lemonchiffon",
    "savefig.facecolor": "auto",
    "axes.titleweight": "bold",
    "grid.alpha": 0.3
})

# files corresponding to our trained agents
MODEL_FILES = {
    "Kuhn": {
        "CFR": "cfr.json",
        "CFR+": "cfr_plus.json",
        "Q-learning Random": "q_learning_random_decay_alpha.json",
        "Q-learning Mixed": "q_learning_mixed_decay_alpha.json",
        "DQN Random Huber": "dqn_random.json",
        "DQN Mixed Huber": "dqn_mixed.json",
        "DQN Mixed MSE": "dqn_mixed_mse.json"
    },
    "Leduc": {
        "CFR": "cfr.json",
        "CFR+": "cfr_plus.json",
        "Q-learning Random": "q_learning_random_decay_alpha.json",
        "Q-learning Mixed": "q_learning_mixed_decay_alpha.json",
        "DQN Random": "dqn_random_mse.json",
        "DQN Mixed": "dqn_mixed_mse.json"
    }
}

# configure the application before rendering any streamlit elements
st.set_page_config(
    page_title="PokerAI",
    page_icon="♠",
    layout="wide"
)

# custom styling for the interface and poker playing cards
st.markdown("""
<style>
.block-container {
    max-width: 1250px;
    padding-top: 2rem;
}
.poker-table {
    background: radial-gradient(ellipse at center, #236651, #103b33);
    border: 9px solid #493626;
    border-radius: 75px;
    padding: 24px;
    margin: 15px 0;
    text-align: center;
    color: #ffffff;
}
.poker-card {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background: #ffffff;
    color: #1d2733;
    width: 78px;
    height: 106px;
    border-radius: 11px;
    border: 2px solid #e0e0e0;
    box-shadow: 0 4px 10px rgba(0,0,0,0.25);
    font-size: 34px;
    font-weight: 800;
    margin: 6px;
}
.poker-card.hidden {
    background: repeating-linear-gradient(
        45deg, #543892, #543892 7px, #7553b6 7px, #7553b6 14px
    );
    border: 3px solid #ffffff;
    color: #ffffff;
}
.table-label {
    color: #d5e6df;
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 6px;
}
.pot-label {
    font-size: 23px;
    font-weight: 700;
    color: #f3d68d;
    margin: 12px 0;
}
</style>
""", unsafe_allow_html=True)


# load only policies that exist on disk
# a trained neural network is not required because its policy is saved as json
def available_policies(game_name):
    policies = {}

    if game_name == "Kuhn":
        # the analytical nash policy is defined directly in the repository
        policies["Nash"] = NashAgent().strategy
        policies["Passive"] = kuhn_baseline()
        policies["Random"] = kuhn_baseline(random_play=True)

    else:
        # leduc does not have a separately saved exact nash model
        policies["Passive"] = leduc_baseline("Passive")
        policies["Random"] = leduc_baseline("Random")

    # load the trained strategies corresponding to this game
    folder = ROOT / "models" / game_name.lower()

    for name, filename in MODEL_FILES[game_name].items():
        path = folder / filename

        if path.is_file():
            try:
                policies[name] = load_strategy(path)
            except (OSError, ValueError) as error:
                st.warning(f"Could not load {filename}: {error}")

    return policies


# read experiment results from their saved csv files
def read_results(path):
    if not path.is_file():
        return []

    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


# create a new game using the selected rules
def create_game(game_name):
    return KuhnPoker() if game_name == "Kuhn" else LeducPoker()


# calculate the current pot using the existing game implementation
def current_pot(game):
    if isinstance(game, LeducPoker):
        return game.pot

    # kuhn starts with two antes and adds one chip per bet or call
    return 2 + game.history.count("b") + game.history.count("k")


# create a game session and preserve results across multiple hands
def new_session(game_name, opponent_name, seat, policy):
    return {
        "configuration": (game_name, opponent_name, seat),
        "game": create_game(game_name),
        "opponent": StrategyAgent(policy),
        "moves": [],
        "results": [],
        "settled": False
    }


# automatically play until the human gets a turn or the hand finishes
def advance_opponent(session, human_seat):
    game = session["game"]

    while not game.done and game.current_player != human_seat:
        info_set = game.get_information_set()
        actions = game.legal_actions()

        # select an action using the opponents saved strategy
        action = session["opponent"].choose_action(
            info_set, actions
        )

        # record the move before applying it to the game
        round_number = (
            game.round_index + 1
            if isinstance(game, LeducPoker)
            else 1
        )

        session["moves"].append(
            f"AI · Round {round_number}: {action.capitalize()}"
        )
        game.step(action)


# record a completed hand exactly once
def settle_hand(session, human_seat):
    game = session["game"]

    if game.done and not session["settled"]:
        payoff = game.get_payoff()[human_seat]

        session["results"].append({
            "hand": len(session["results"]) + 1,
            "payoff": payoff,
            "history": game.history
        })

        session["settled"] = True


# draw a simple playing card using html and css
def card_html(rank=None):
    if rank is None:
        return '<div class="poker-card hidden">♠</div>'

    return f'<div class="poker-card">{escape(str(rank))}</div>'


# show the current cards, public card and pot
def render_table(game, human_seat):
    opponent_seat = 1 - human_seat
    opponent_card = (
        game.cards[opponent_seat] if game.done else None
    )

    public_card = (
        game.public_card
        if isinstance(game, LeducPoker)
        else None
    )

    # the public card remains hidden until the second leduc betting round
    board_html = (
        card_html(public_card)
        if isinstance(game, LeducPoker)
        else '<span style="color:#d5e6df">No public card</span>'
    )

    st.markdown(
        f"""
        <div class="poker-table">
            <div class="table-label">AI Opponent</div>
            {card_html(opponent_card)}
            <div class="pot-label">Pot: {current_pot(game)} chips</div>
            <div class="table-label">Community Card</div>
            {board_html}
            <div style="height:16px"></div>
            <div class="table-label">Your Hand</div>
            {card_html(game.cards[human_seat])}
        </div>
        """,
        unsafe_allow_html=True
    )


# plot cumulative net profit from all completed hands
def plot_bankroll(results):
    profits = [row["payoff"] for row in results]
    cumulative = [0, *accumulate(profits)]
    hands = list(range(len(cumulative)))

    fig, ax = plt.subplots(figsize=(9, 3.3))

    ax.plot(
        hands,
        cumulative,
        marker="o",
        markersize=3,
        color="#425e87",
        linewidth=2
    )

    ax.axhline(0, linestyle="--", color="#555555")
    ax.set_xlabel("Completed hands")
    ax.set_ylabel("Cumulative profit (chips)")
    ax.set_title("Your Poker P&L")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# let the user play against one of the trained agents
def play_page(game_name, policies):
    st.title("♠ Play Poker")
    st.caption("Challenge trained AI agents in Kuhn or Leduc Poker.")

    trained = [
        name for name in policies
        if name not in ("Passive", "Random")
    ]
    options = trained + ["Passive", "Random"]

    if not options:
        st.error("No opponents are available.")
        return

    first, second = st.columns(2)

    with first:
        opponent_name = st.selectbox(
            "Choose your opponent",
            options,
            key=f"opponent_{game_name}"
        )

    with second:
        human_seat = st.radio(
            "Your position",
            (0, 1),
            format_func=lambda x: f"Player {x}",
            horizontal=True,
            key=f"seat_{game_name}"
        )

    configuration = (game_name, opponent_name, human_seat)

    # reset the session when the game, opponent or player position changes
    if (
        "poker_session" not in st.session_state
        or st.session_state["poker_session"]["configuration"] != configuration
    ):
        st.session_state["poker_session"] = new_session(
            game_name,
            opponent_name,
            human_seat,
            policies[opponent_name]
        )

    session = st.session_state["poker_session"]
    game = session["game"]

    # allow the opponent to act first when the human is player 1
    advance_opponent(session, human_seat)
    settle_hand(session, human_seat)

    render_table(game, human_seat)

    # show the betting round and publicly visible history
    if isinstance(game, LeducPoker):
        st.caption(
            f"Betting round: {game.round_index + 1} of 2 "
            f"· History: {game.history or 'No actions'}"
        )
    else:
        st.caption(f"History: {game.history or 'No actions'}")

    if game.done:
        payoff = game.get_payoff()[human_seat]

        if payoff > 0:
            st.success(f"You won {payoff} chips!")
        elif payoff < 0:
            st.error(f"You lost {abs(payoff)} chips.")
        else:
            st.info("The hand ended in a tie.")

    else:
        st.subheader("Your turn")

        legal_actions = game.legal_actions()
        action_columns = st.columns(len(legal_actions))

        for column, action in zip(action_columns, legal_actions):
            with column:
                if st.button(
                    action.capitalize(),
                    key=f"action_{action}",
                    use_container_width=True
                ):
                    # record the human decision before the game state changes
                    round_number = (
                        game.round_index + 1
                        if isinstance(game, LeducPoker)
                        else 1
                    )

                    session["moves"].append(
                        f"You · Round {round_number}: {action.capitalize()}"
                    )

                    game.step(action)
                    advance_opponent(session, human_seat)
                    settle_hand(session, human_seat)
                    st.rerun()

    # starting a new hand keeps the overall results and opponent settings
    if st.button(
        "Deal next hand" if game.done else "Deal new hand",
        type="primary"
    ):
        session["game"] = create_game(game_name)
        session["moves"] = []
        session["settled"] = False
        st.rerun()

    with st.expander("Hand action history", expanded=False):
        if session["moves"]:
            for move in session["moves"]:
                st.write(move)
        else:
            st.caption("No actions have been taken yet.")

    st.divider()
    st.subheader("Session performance")

    results = session["results"]
    total_profit = sum(row["payoff"] for row in results)
    wins = sum(row["payoff"] > 0 for row in results)
    losses = sum(row["payoff"] < 0 for row in results)

    metrics = st.columns(4)
    metrics[0].metric("Hands completed", len(results))
    metrics[1].metric("Wins", wins)
    metrics[2].metric("Losses", losses)
    metrics[3].metric("Net P&L", f"{total_profit:+} chips")

    if results:
        plot_bankroll(results)

        st.dataframe(
            list(reversed(results)),
            hide_index=True,
            use_container_width=True
        )


# plot the head to head matrix using exact expected payoffs
def plot_head_to_head(rows):
    names = [row["agent"] for row in rows]

    # columns have the same agent labels as the matrix rows
    values = [
        [float(row[name]) for name in names]
        for row in rows
    ]

    maximum = max(
        0.1,
        max(abs(value) for line in values for value in line)
    )

    fig, ax = plt.subplots(figsize=(10, 7))

    heatmap = ax.imshow(
        values,
        cmap="PuOr",
        vmin=-maximum,
        vmax=maximum
    )

    ax.set_xticks(
        range(len(names)), names,
        rotation=35, ha="right"
    )
    ax.set_yticks(range(len(names)), names)

    ax.set_xlabel("Opponent")
    ax.set_ylabel("Agent")
    ax.set_title("Exact Head-to-Head Expected Payoff")

    # print expected payoff inside each matrix cell
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(
                j, i,
                f"{values[i][j]:+.2f}",
                ha="center",
                va="center",
                fontsize=8
            )

    fig.colorbar(heatmap, ax=ax, label="Chips per hand")
    fig.tight_layout()

    st.pyplot(fig)
    plt.close(fig)


# plot the convergence of cfr and cfr+ from our stored experiments
def plot_cfr_convergence(rows):
    fig, ax = plt.subplots(figsize=(9, 4))

    for name, color in (
        ("CFR", "#425e87"),
        ("CFR+", "#7051a8")
    ):
        subset = [
            row for row in rows
            if row["algorithm"] == name
        ]

        if subset:
            ax.plot(
                [int(row["iterations"]) for row in subset],
                [float(row["exploitability"]) for row in subset],
                marker="o",
                markersize=4,
                linewidth=2,
                color=color,
                label=name
            )

    ax.set_xlabel("Training iterations")
    ax.set_ylabel("Exploitability")
    ax.set_title("CFR Convergence")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# display the results of our existing research experiments
def research_page(game_name):
    st.title("Research Dashboard")
    st.caption(
        "Compare expected profit, exploitability and statistical "
        "variation using our saved experimental results."
    )

    folder = ROOT / "results" / game_name.lower()

    summary_file = (
        "agent_summary.csv"
        if game_name == "Leduc"
        else "kuhn_final_benchmark.csv"
    )

    rows = read_results(folder / summary_file)

    if rows:
        # present the most important research measurements
        best_robust = min(
            rows, key=lambda row: float(row["exploitability"])
        )
        best_random = max(
            rows, key=lambda row: float(row["ev_vs_random"])
        )

        first, second, third = st.columns(3)

        first.metric("Strategies evaluated", len(rows))
        second.metric(
            "Lowest exploitability",
            f"{float(best_robust['exploitability']):.4f}"
        )
        third.metric(
            "Highest EV vs Random",
            f"{float(best_random['ev_vs_random']):+.4f}"
        )

        st.caption(
            f"Lowest exploitability: {best_robust['agent']} "
            f"· Best vs Random: {best_random['agent']}"
        )

        st.subheader("Algorithm performance")
        st.dataframe(
            rows,
            hide_index=True,
            use_container_width=True
        )

    else:
        st.info(
            f"No final benchmark found at {folder / summary_file}. "
            "Run the saved benchmark experiment to create it."
        )

    # leduc has a six agent square payoff matrix
    matrix_path = folder / "head_to_head_matrix.csv"

    if game_name == "Leduc" and matrix_path.is_file():
        st.subheader("Head-to-head comparison")
        plot_head_to_head(read_results(matrix_path))
        st.caption(
            "Each cell is the balanced exact EV for the row agent "
            "against the column agent. Positive means the row agent profits."
        )

    # display convergence curves for the equilibrium oriented algorithms
    convergence = read_results(folder / "cfr_comparison.csv")

    if convergence:
        st.subheader("Equilibrium convergence")
        plot_cfr_convergence(convergence)

    # reuse the monte carlo figures rather than repeating the simulations
    if game_name == "Leduc":
        overview = folder / "monte_carlo_comparison.png"
        variance = folder / "monte_carlo_cfr_variance.png"

        if overview.is_file():
            st.subheader("Monte Carlo validation")
            st.image(str(overview), use_container_width=True)
            st.caption(
                "Simulated mean EV and 95% confidence intervals "
                "compared with exact expected payoff."
            )

        if variance.is_file():
            st.subheader("Finite-sample variation")
            st.image(str(variance), use_container_width=True)

    st.caption(
        "Exact EV uses enumeration rather than sampling. "
        "Monte Carlo simulations illustrate finite-sample variation."
    )


# explore the action probabilities of different trained policies
def strategy_page(game_name, policies):
    st.title("Strategy Explorer")
    st.caption(
        "Inspect how the trained agents act given the information "
        "visible to the player."
    )

    if game_name == "Kuhn":
        # kuhn decisions depend on the private card and betting history
        card = st.selectbox("Your private card", ("J", "Q", "K"))
        history = st.selectbox(
            "Betting history",
            ("", "c", "b", "cb"),
            format_func=lambda value: value or "Start of hand"
        )

        info = f"{card}|{history}"
        actions = (
            ("check", "bet")
            if history in ("", "c")
            else ("call", "fold")
        )

    else:
        # use the evaluator to obtain only valid leduc information sets
        records = leduc_information_sets()

        card = st.selectbox("Your private card", ("J", "Q", "K"))
        board = st.selectbox(
            "Public card",
            ("-", "J", "Q", "K"),
            format_func=lambda value: (
                "Not revealed" if value == "-" else value
            )
        )

        matching = [
            (info, legal)
            for info, _, legal in records
            if info.split("|")[0] == card
            and info.split("|")[1] == board
        ]

        options = sorted(
            (info for info, _ in matching),
            key=lambda value: (len(value), value)
        )

        info = st.selectbox(
            "Betting history",
            options,
            format_func=lambda value: (
                value.split("|", 2)[2] or "Start of round"
            )
        )

        actions = next(
            legal for state, legal in matching
            if state == info
        )

    st.code(info, language="text")

    # build a comparison table across every available trained policy
    comparison = []

    for name, strategy in policies.items():
        probabilities = strategy.get(info)

        if probabilities is None:
            continue

        comparison.append({
            "Agent": name,
            **{
                action.capitalize(): f"{probabilities[action]:.1%}"
                for action in actions
            }
        })

    st.subheader("Action probabilities")
    st.dataframe(
        comparison,
        hide_index=True,
        use_container_width=True
    )

    # show a more detailed graph for one selected strategy
    selected = st.selectbox(
        "Inspect a specific agent",
        list(policies)
    )

    distribution = policies[selected].get(info)

    if distribution is not None:
        fig, ax = plt.subplots(figsize=(7, 3.5))

        ax.bar(
            [action.capitalize() for action in actions],
            [distribution[action] for action in actions],
            color="#7051a8"
        )

        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Probability")
        ax.set_title(f"{selected}: Action Distribution")
        ax.grid(axis="y", alpha=0.3)

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.caption(
        "The saved Q-learning and DQN policies use greedy action "
        "selection. CFR policies may mix actions probabilistically. "
        "These probabilities describe behaviour, not the internal "
        "reasoning of the neural network."
    )


# application navigation and game selection
st.sidebar.title("♠ PokerAI")
st.sidebar.caption("")

page = st.sidebar.radio(
    "Navigation",
    ("Play Poker", "Research Dashboard", "Strategy Explorer")
)

game_name = st.sidebar.selectbox(
    "Poker game",
    ("Kuhn", "Leduc")
)

st.sidebar.divider()

policies = available_policies(game_name)

if page == "Play Poker":
    play_page(game_name, policies)
elif page == "Research Dashboard":
    research_page(game_name)
else:
    strategy_page(game_name, policies)
