import random

from poker_ai.games.kuhn import KuhnPoker
from poker_ai.evaluation.evaluate import exact_ev


# simulates multiple poker hands between two agents and records the results
def play_match(agent_a, agent_b, n_hands=1000, seed=42):
    # require an even number of hands so both agents play each position equally
    if n_hands <= 0 or n_hands % 2 != 0:
        raise ValueError("n_hands must be a positive even number")

    # separate rng so we can reproduce the same card deals
    rng = random.Random(seed)

    # track total profit/loss and number of wins and losses for agent A
    total_payoff_a = 0
    wins_a = 0
    losses_a = 0

    # play each card deal twice with the agents swapping positions
    for _ in range(n_hands // 2):
        cards = rng.sample(["J", "Q", "K"], 2)

        # agent A plays as player 0 first, then player 1
        for a_is_player0 in (True, False):
            if a_is_player0:
                agents = (agent_a, agent_b)
            else:
                agents = (agent_b, agent_a)

            # start a new hand using the same card deal
            game = KuhnPoker()
            game.cards = cards.copy()

            # keep playing until the game finishes
            while not game.done:
                player = game.current_player

                # ask the current agent to choose an action using its strategy
                action = agents[player].choose_action(
                    game.get_information_set(),
                    game.legal_actions()
                )

                # update the game based on the chosen action
                game.step(action)

            # get the payoff for agent A regardless of which position it played
            payoff_a = game.get_payoff()[
                0 if a_is_player0 else 1
            ]

            # update the total profit/loss and number of wins or losses
            total_payoff_a += payoff_a

            if payoff_a > 0:
                wins_a += 1
            elif payoff_a < 0:
                losses_a += 1

    # return the match statistics from the perspective of agent A
    return {
        "hands": n_hands,
        "wins_a": wins_a,
        "losses_a": losses_a,
        "win_rate_a": wins_a / n_hands,
        "total_payoff_a": total_payoff_a,
        "avg_payoff_a": total_payoff_a / n_hands,
    }


# calculates the exact expected payoff between two strategies across both positions
def balanced_exact_ev(strategy_a, strategy_b):
    # calculate the expected payoff when A is player 0
    ev_a_first = exact_ev(strategy_a, strategy_b)[0]

    # calculate the expected payoff when A is player 1
    ev_a_second = exact_ev(strategy_b, strategy_a)[1]

    # average both positions to remove the advantage of playing in a specific seat
    return (ev_a_first + ev_a_second) / 2
