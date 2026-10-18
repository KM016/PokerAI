
import random

from poker_ai.games.kuhn import KuhnPoker
from poker_ai.agents.nash_agent import NashAgent

# monte carlo estimate of player 0's average earnings
def simulate(n_hands=100_000, seed=42):
    # make the random card dealing reproducible
    random.seed(seed)

    # create the poker game and two Nash agents
    game = KuhnPoker()
    agents = {
        0: NashAgent(seed=seed),
        1: NashAgent(seed=seed + 1)
    }

    # track player 0's total profit
    total_profit = 0

    # play the specified number of hands
    for _ in range(n_hands):
        game.reset()

        # continue until the current hand finishes
        while not game.done:
            player = game.current_player

            # give the agent only its own information
            info = game.get_information_set()
            legal = game.legal_actions()

            # agent chooses an action and the game executes it
            action = agents[player].choose_action(info, legal)
            game.step(action)

        # record player 0's profit for this hand
        payoff_0, payoff_1 = game.get_payoff()
        total_profit += payoff_0

    # calculate average profit per hand
    average_profit = total_profit / n_hands

    return average_profit


if __name__ == "__main__":
    n_hands = 1_000_000
    simulated_ev = simulate(n_hands)
    theoretical_ev = -1 / 18

    print(f"Hands played: {n_hands:,}")
    print(f"Simulated Player 0 EV: {simulated_ev:.5f}")
    print(f"Theoretical Player 0 EV: {theoretical_ev:.5f}")
    print(f"Absolute error: {abs(simulated_ev - theoretical_ev):.5f}")
