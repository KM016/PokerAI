import math
from collections import Counter, defaultdict
from copy import copy
from functools import lru_cache
from itertools import permutations

from poker_ai.games.leduc import LeducPoker


# generate all possible card deals and calculate their probabilities
# we use these to evaluate strategies across every possible combination of cards
# rather than relying on randomly simulated games
@lru_cache(maxsize=1)
def possible_deals():
    # leduc has 6 physical cards and we deal 3 without replacement
    # 1 private card for each player and 1 shared public card
    # this gives 6 * 5 * 4 = 120 possible ordered physical deals
    # permutations generates every possible ordering of 3 cards
    # counter groups deals that have the same ranks because the deck has duplicates
    counts = Counter(permutations(LeducPoker.DECK, 3))

    # count how many physical deals exist in total
    # this equals 120 and is used to convert frequencies into probabilities
    total = sum(counts.values())

    # return each distinct deal alongside its probability
    # for example some rank combinations can occur more often than others
    # there are 24 distinct rank deals after combining identical physical cards
    # sorted makes the ordering consistent whenever we call the function
    # tuples are used because the result does not need to change
    return tuple(
        (deal, count / total)
        for deal, count in sorted(counts.items())
    )


# create a new game state after taking a particular action
# we need to do this because evaluation explores multiple possible decisions
# and each decision must have its own independent game state
def next_game(game, action):
    # make a shallow copy of the current game
    # this copies the game object without copying all its contents recursively
    new_game = copy(game)

    # lists are mutable so they must be copied separately
    # otherwise changing one game branch could also change the original game
    new_game.cards = game.cards.copy()
    new_game.contributions = game.contributions.copy()
    new_game.round_contributions = game.round_contributions.copy()

    # apply the chosen action to the copied game
    # step updates the betting history, pot, current player and other game variables
    # the original game remains unchanged so we can explore other actions
    new_game.step(action)

    # return the new game state after taking the action
    return new_game


# identify every information set that a player could encounter in leduc
# an information set describes everything the acting player knows at that moment
# this includes their private card, the public card and the betting history
# we also record which player acts and which actions are legal
@lru_cache(maxsize=1)
def information_sets():
    # dictionary used to store each unique information set
    # the information set string is the key
    # the value contains the acting player and available actions
    found = {}

    # recursively explore every possible decision from a given game state
    def explore(game):
        # stop exploring once the hand has finished
        # terminal states have no further decisions or information sets
        if game.done:
            return

        # get the information visible to the player whose turn it is
        info = game.get_information_set()

        # save the current information set and its legal actions
        # tuple converts the actions into an immutable sequence
        # using a dictionary means repeated visits to the same information set
        # will update the same entry instead of creating duplicates
        found[info] = (
            game.current_player,
            tuple(game.legal_actions())
        )

        # explore what happens after every action the player could take
        # each action produces another game state which we explore recursively
        # this continues until every possible betting sequence has been explored
        for action in game.legal_actions():
            explore(next_game(game, action))

    # repeat the exploration for every possible combination of cards
    # the deal probabilities are not needed here because we only want the states
    # the underscore means we are intentionally ignoring the probability
    for deal, _ in possible_deals():
        explore(LeducPoker(deal=deal))

    # convert the dictionary into a tuple of information set records
    # each record contains the information set, acting player and legal actions
    # sorted ensures that the result always has the same ordering
    return tuple(
        (info, player, actions)
        for info, (player, actions) in sorted(found.items())
    )


# validate the action probabilities stored in a strategy
# a strategy assigns a probability to each available action at an information set
# for example check = 0.7 and bet = 0.3
# every legal action must be included and probabilities must sum to 1
def action_probabilities(strategy, game):
    # get the information set for the current player
    info = game.get_information_set()

    # get the actions the player is actually allowed to take
    legal = game.legal_actions()

    # check that the strategy contains a decision rule for this information set
    # a missing information set means the strategy is incomplete
    if info not in strategy:
        raise ValueError(f"Missing information set: {info}")

    # retrieve the action probabilities from the strategy
    probabilities = strategy[info]

    # ensure the strategy contains exactly the actions allowed by the game
    # converting both collections to sets ignores the order of the actions
    if set(probabilities) != set(legal):
        raise ValueError(f"Incorrect actions at {info}")

    # extract the probability assigned to each legal action
    # this gives us a list of numerical probabilities to validate
    values = [probabilities[action] for action in legal]

    # reject probabilities that are not valid numerical values
    # isinstance checks that the value is a number
    # isfinite rejects infinity and nan values
    # probabilities must also be nonnegative
    if any(
        not isinstance(p, (int, float))
        or not math.isfinite(p)
        or p < 0
        for p in values
    ):
        raise ValueError(f"Invalid probabilities at {info}")

    # probabilities across all available actions must add up to 1
    # the small tolerance allows for floating point rounding errors
    if abs(sum(values) - 1.0) > 1e-6:
        raise ValueError(f"Probabilities do not sum to one at {info}")

    # return the validated probability distribution
    return probabilities


# calculate the expected payoff from a particular game state
# both players follow their supplied strategies for the rest of the game
# we explore every possible action and weight its payoff by its probability
def expected_payoff(game, strategies):
    # if the game has finished there are no more decisions to consider
    # return the final net profit or loss for both players
    if game.done:
        return game.get_payoff()

    # identify which player is making the next decision
    player = game.current_player

    # retrieve the action probabilities from that players strategy
    # strategies[0] belongs to player 0 and strategies[1] belongs to player 1
    probabilities = action_probabilities(strategies[player], game)

    # store the expected payoff for each player
    # both values start at zero because we have not explored any actions yet
    ev = [0.0, 0.0]

    # calculate the payoff from every action the current player could take
    for action in game.legal_actions():
        # probability that the current player chooses this particular action
        probability = probabilities[action]

        # an action with zero probability cannot contribute to expected payoff
        # skip it to avoid exploring an unnecessary branch of the game tree
        if probability == 0:
            continue

        # take the action and recursively calculate the expected future payoff
        # this returns the expected profit or loss for both players
        future = expected_payoff(
            next_game(game, action),
            strategies
        )

        # add this actions weighted payoff to the running expected value
        # probability * future payoff gives the contribution from this action
        # we calculate the expected payoff separately for players 0 and 1
        for p in (0, 1):
            ev[p] += probability * future[p]

    # return the expected profit or loss for both players from this game state
    return tuple(ev)


# calculate the exact expected payoff when two strategies play against each other
# the result averages over every possible card deal and every possible action
# unlike monte carlo simulation there is no random sampling error
def exact_ev(strategy_0, strategy_1):
    # store the overall expected payoff for both players
    result = [0.0, 0.0]

    # consider every possible card deal along with its probability
    for deal, probability in possible_deals():
        # create a new game using this specific combination of cards
        game = LeducPoker(deal=deal)

        # calculate the expected payoff when both players follow their strategies
        # this considers every possible action sequence for the given deal
        payoff = expected_payoff(
            game,
            (strategy_0, strategy_1)
        )

        # weight the payoff by how likely the card deal is to occur
        # then add it to the overall expected payoff for each player
        # mathematically this is the law of total expectation
        for player in (0, 1):
            result[player] += probability * payoff[player]

    # return the expected net profit or loss for both players
    return tuple(result)


# evaluate a strategy against an opponent from both player positions
# this is useful because acting first can affect performance in poker
# averaging both seats provides a fairer comparison between strategies
def balanced_exact_ev(strategy, opponent):
    # evaluate our strategy as player 0 against the opponent as player 1
    # the first element gives the expected payoff belonging to our strategy
    first = exact_ev(strategy, opponent)[0]

    # swap positions so the opponent is player 0 and our strategy is player 1
    # the second element now gives the expected payoff belonging to our strategy
    second = exact_ev(opponent, strategy)[1]

    # average performance from both positions to reduce seat bias
    return (first + second) / 2


# construct simple opponents that we can use as reference strategies
# these are not trained agents and their decisions are determined by fixed rules
# passive avoids aggression, random selects actions uniformly
# rule based uses basic card strength information to make decisions
def baseline_strategy(style):
    # check that the selected opponent type is supported
    if style not in ("Passive", "Random", "Rule-based"):
        raise ValueError("Unknown baseline strategy")

    # store the action probabilities for every information set
    strategy = {}

    # go through every information set that can occur in leduc
    # we do not need the player number here so we ignore it with an underscore
    for info, _, actions in information_sets():
        # separate the information set into its individual components
        # card is the private card, board is the shared card
        # the third component is the betting history which we do not need here
        card, board, _ = info.split("|", 2)

        if style == "Random":
            # the random opponent gives every available action equal probability
            # example: with 3 legal actions each is selected with probability 1/3
            strategy[info] = {
                action: 1 / len(actions)
                for action in actions
            }

            # this information set is complete so move to the next one
            continue

        if style == "Passive":
            # passive always checks when checking is possible
            # if facing a bet it folds rather than calling or raising
            # this opponent therefore never voluntarily adds betting chips
            chosen = "check" if "check" in actions else "fold"

        else:
            # rule based uses a simple heuristic based on the players cards
            # this is not optimal poker play or a nash equilibrium strategy

            # a pair occurs when the private card matches the public card
            # the public card must also have been revealed
            # a pair is the strongest possible type of hand in leduc
            pair = board != "-" and card == board

            # if there is no outstanding bet we can either check or bet
            # bet when holding a pair or a king and otherwise check
            if "check" in actions:
                chosen = "bet" if pair or card == "K" else "check"

            # when facing a bet with a pair we raise if raising is allowed
            elif pair and "raise" in actions:
                chosen = "raise"

            # otherwise call with a pair, a king or a queen on a non king board
            # this is a simple rule based on relative hand strength
            elif pair or card == "K" or (card == "Q" and board != "K"):
                chosen = "call"

            # fold weaker hands when facing an outstanding bet
            else:
                chosen = "fold"

        # convert the selected action into a probability distribution
        # the chosen action receives probability 1 and every other action gets 0
        # float converts true to 1.0 and false to 0.0
        strategy[info] = {
            action: float(action == chosen)
            for action in actions
        }

    # return a complete strategy covering every possible information set
    return strategy


# calculate the expected payoff when one player follows a fixed best response
# chosen_actions contains the decisions already selected for the responding player
# the opponent continues following its original strategy
def response_payoff(game, player, opponent_strategy, chosen_actions):
    # if the game is finished return the payoff belonging to the responder
    # unlike expected_payoff we only need one players payoff here
    if game.done:
        return game.get_payoff()[player]

    # check whether it is the responding players turn to act
    if game.current_player == player:
        # get the information available to the responding player
        info = game.get_information_set()

        # look up the action previously selected for this information set
        # the same action is used for every hidden state in that information set
        action = chosen_actions[info]

        # take the selected action and calculate the payoff from the next state
        # the responding player is following a deterministic strategy here
        return response_payoff(
            next_game(game, action),
            player,
            opponent_strategy,
            chosen_actions
        )

    # if it is the opponents turn then they follow their original strategy
    # retrieve the probability assigned to each of their legal actions
    probabilities = action_probabilities(opponent_strategy, game)

    # calculate the expected payoff across every action the opponent could take
    # each future payoff is multiplied by the probability of that action
    # zero probability actions are ignored because they cannot occur
    return sum(
        probability * response_payoff(
            next_game(game, action),
            player,
            opponent_strategy,
            chosen_actions
        )
        for action, probability in probabilities.items()
        if probability > 0
    )


# find the most profitable possible strategy against a fixed opponent
# this is called the best response and is used to calculate exploitability
# unlike a normal strategy the best response is optimised for one opponent
# it must still respect imperfect information and cannot observe hidden cards
def best_response(opponent_strategy, player):
    # check which player we are calculating the best response for
    # player 0 and player 1 are the only valid positions
    if player not in (0, 1):
        raise ValueError("Player must be 0 or 1")

    # group all game states that belong to the same information set
    # different hidden card deals may lead to exactly the same observations
    # the responder must make the same decision across these hidden states
    # defaultdict(list) creates an empty list automatically for a new key
    nodes = defaultdict(list)

    # recursively collect game states where the responding player must act
    # reach represents the counterfactual probability of reaching that state
    # it includes chance and opponent actions but excludes our own actions
    def collect(game, reach):
        # stop at completed games or branches with zero reach probability
        if game.done or reach == 0:
            return

        # get the information set for the player whose turn it is
        info = game.get_information_set()

        if game.current_player == player:
            # the responding player is making the next decision
            # save the underlying game state and its counterfactual reach
            # multiple states can share the same information set due to hidden cards
            nodes[info].append((game, reach))

            # explore every legal response action
            # do not multiply reach by the responders action probabilities
            # we are searching for an optimal policy rather than following one
            # this allows us to evaluate all possible decisions independently
            for action in game.legal_actions():
                collect(next_game(game, action), reach)

        else:
            # the opponent is making the next decision
            # unlike the responder the opponent follows a fixed strategy
            probabilities = action_probabilities(opponent_strategy, game)

            # explore each possible opponent action
            # multiply reach by its probability because opponent choices
            # affect how likely the resulting game state is to occur
            for action, probability in probabilities.items():
                if probability > 0:
                    collect(
                        next_game(game, action),
                        reach * probability
                    )

    # start exploring from every possible card deal
    # each deal begins with its own chance probability
    # this ensures that more likely card combinations receive greater weight
    for deal, probability in possible_deals():
        collect(LeducPoker(deal=deal), probability)

    # store the optimal action selected for each information set
    # these decisions will be calculated starting from the end of the game
    chosen_actions = {}

    # arrange information sets so the deepest decisions are evaluated first
    # split extracts the betting history from the information set string
    # the length of this history measures how many actions have occurred
    # reverse true means longer histories appear before shorter histories
    # this allows us to solve future decisions before earlier decisions
    ordered = sorted(
        nodes,
        key=lambda info: len(info.split("|", 2)[2]),
        reverse=True
    )

    # calculate the best response action at each information set
    for info in ordered:
        # get the legal actions from one of the game states in this information set
        # every state in the same information set has the same legal actions
        legal = nodes[info][0][0].legal_actions()

        # store the expected value of choosing each possible action
        action_values = {}

        for action in legal:
            # calculate the value of taking this action across every hidden state
            # the responder cannot choose a different action depending on
            # the opponents hidden card so we aggregate all these states together
            # each state is weighted by its counterfactual reach probability

            # after taking the action we use already solved future decisions
            # response_payoff follows the fixed opponent strategy and our
            # chosen actions at later information sets
            action_values[action] = sum(
                reach * response_payoff(
                    next_game(game, action),
                    player,
                    opponent_strategy,
                    chosen_actions
                )
                for game, reach in nodes[info]
            )

        # select the action with the highest total counterfactual value
        # max compares all legal actions using their calculated expected payoffs
        # the result is stored as the optimal decision for this information set
        chosen_actions[info] = max(
            legal,
            key=lambda action: action_values[action]
        )

    # create the final best response strategy in the usual probability format
    # this allows us to evaluate it using the same functions as other agents
    response = {}

    # go through every information set belonging to the responding player
    for info, acting_player, legal in information_sets():
        # skip information sets belonging to the opponent
        if acting_player != player:
            continue

        # retrieve the best action we calculated for this information set
        # some information sets may be unreachable against this opponent
        # in that case use the first legal action as a valid default
        chosen = chosen_actions.get(info, legal[0])

        # convert the chosen action into a deterministic strategy
        # probability 1 means the action is always selected
        # probability 0 means the action is never selected
        response[info] = {
            action: float(action == chosen)
            for action in legal
        }

    # calculate the exact expected profit earned by the best response
    # we must place the best response strategy in the correct player position
    if player == 0:
        # best response plays as player 0 against the opponent as player 1
        value = exact_ev(response, opponent_strategy)[0]
    else:
        # opponent plays as player 0 and best response plays as player 1
        value = exact_ev(opponent_strategy, response)[1]

    # return both the optimal expected profit and the strategy that achieves it
    return value, response


# calculate how exploitable a strategy is against an optimal opponent
# an exploitable strategy allows an opponent to earn extra expected profit
# lower exploitability means a strategy is closer to a nash equilibrium
def exploitability(strategy):
    # calculate the optimal expected payoff against this strategy as player 0
    br0, _ = best_response(strategy, player=0)

    # repeat with the best response acting as player 1
    br1, _ = best_response(strategy, player=1)

    # average the best response values from both player positions
    # in a two player zero sum game a nash equilibrium has exploitability zero
    # this gives us a measure of how vulnerable the supplied strategy is
    return (br0 + br1) / 2
