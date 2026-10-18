
from copy import deepcopy
from itertools import permutations, product

from poker_ai.games.kuhn import KuhnPoker
from poker_ai.agents.nash_agent import NashAgent

# mathematically calculates expected profit by considering all possible outcomes 
def expected_payoff(game, strategies):
    # if game has ended then return actual payouts
    if game.done:
        return game.get_payoff()

    # find the current player's strategy
    player = game.current_player
    info = game.get_information_set()
    probabilities = strategies[player][info]

    # track both players' expected payouts
    ev = [0.0, 0.0]

    # explore all possible legal action
    for action in game.legal_actions():
        probability = probabilities[action]

        # copy the game so each action is explored separately
        next_game = deepcopy(game)
        next_game.step(action)

        # calculate expected payouts after this action
        next_ev = expected_payoff(next_game, strategies)

        # weight the payouts by this action's probability
        ev[0] += probability * next_ev[0]
        ev[1] += probability * next_ev[1]

    return tuple(ev)


def exact_ev(strategy_0, strategy_1):
    # all 6 possible ways to deal 2 different cards
    deals = list(permutations(["J", "Q", "K"], 2))

    total_ev = [0.0, 0.0]
    strategies = [strategy_0, strategy_1]

    for cards in deals:
        game = KuhnPoker()
        game.cards = list(cards)

        # calculate EV for this particular card deal
        payoff = expected_payoff(game, strategies)

        # every card deal is equally likely
        total_ev[0] += payoff[0] / len(deals)
        total_ev[1] += payoff[1] / len(deals)

    return tuple(total_ev)


def best_response(opponent_strategy, player):
    # check that were evaluating player 0 or player 1
    if player not in [0, 1]:
        raise ValueError("Player must be 0 or 1")

    # each player has 6 information sets
    if player == 0:
        histories = ["", "cb"]
    else:
        histories = ["c", "b"]

    # generate the 6 information sets for this player
    infos = [
        f"{card}|{history}"
        for history in histories
        for card in ["J", "Q", "K"]
    ]

    # the two possible actions at each information set
    actions = {
        "": ["check", "bet"],
        "c": ["check", "bet"],
        "b": ["call", "fold"],
        "cb": ["call", "fold"]
    }

    # get the legal actions for each information set
    options = [actions[info.split("|")[1]] for info in infos]

    best_ev = float("-inf")
    best_strategy = None

    # generate every possible combination of decisions
    for choices in product(*options):

        # create a strategy that always chooses the selected action
        strategy = {}

        for info, legal, chosen in zip(infos, options, choices):
            strategy[info] = {
                action: float(action == chosen)
                for action in legal
            }

        # calculate its EV against the opponent
        if player == 0:
            ev = exact_ev(strategy, opponent_strategy)[0]
        else:
            ev = exact_ev(opponent_strategy, strategy)[1]

        # keep the strategy with the highest EV
        if ev > best_ev:
            best_ev = ev
            best_strategy = strategy

    return best_ev, best_strategy


def exploitability(strategy):
    # best possible profit against the strategy as player 1
    br0_ev, _ = best_response(strategy, player=0)

    # best possible profit against the strategy as player 0
    br1_ev, _ = best_response(strategy, player=1)

    # average exploitation advantage across both positions
    return (br0_ev + br1_ev) / 2




if __name__ == "__main__":
    agent = NashAgent()

    ev_0, ev_1 = exact_ev(agent.strategy, agent.strategy)

    print(f"Exact Player 0 EV: {ev_0:.6f}")
    print(f"Exact Player 1 EV: {ev_1:.6f}")
    print(f"Theoretical Player 0 EV: {-1/18:.6f}")

    print(f"Nash exploitability: {exploitability(agent.strategy):.8f}")

    br0_ev, _ = best_response(agent.strategy, player=0)
    br1_ev, _ = best_response(agent.strategy, player=1)

    # each game player 0 is disadvantaged as they bet first and this gives player 1 information 
    print(f"Best response as Player 0: {br0_ev:.6f}")
    print(f"Best response as Player 1: {br1_ev:.6f}")
