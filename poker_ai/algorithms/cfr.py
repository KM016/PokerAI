from copy import deepcopy
from itertools import permutations

from poker_ai.games.kuhn import KuhnPoker


# converts cumulative regrets into probabilities for each action
def regret_matching(regrets):
    # negative regrets are treated as 0 -> actions with positive regrets are preferred
    positive_regrets = {
        action: max(regret, 0)
        for action, regret in regrets.items()
    }

    total = sum(positive_regrets.values())

    # if no action has positive regret, choose each action equally often
    if total == 0:
        probability = 1 / len(regrets)
        return {action: probability for action in regrets}

    # actions with higher positive regret get a higher probability
    return {
        action: regret / total
        for action, regret in positive_regrets.items()
    }


# trains a poker strategy using counterfactual regret minimisation (CFR)
class CFRTrainer:
    def __init__(self):
        # stores the total regret for each action at every information set
        self.regret_sum = {}

        # stores the strategies used during training -> used to calculate the average strategy
        self.strategy_sum = {}

        # tracks how many training iterations have been completed
        self.iteration = 0

    def get_strategy(self, info_set, legal_actions):
        # if we haven't encountered this information set before, start with 0 regrets
        if info_set not in self.regret_sum:
            self.regret_sum[info_set] = {
                action: 0.0 for action in legal_actions
            }

            self.strategy_sum[info_set] = {
                action: 0.0 for action in legal_actions
            }

        # use regret matching to decide how often each action should be played
        return regret_matching(self.regret_sum[info_set])


    # explores the game tree and calculates regrets for each possible decision
    def _traverse(self, game, reach, chance, regret_updates, updating_player=None):
        # if the game has finished, return the final profit/loss for both players
        if game.done:
            return game.get_payoff()

        # get whose turn it is, what they know and which actions they can take
        player = game.current_player
        info = game.get_information_set()
        legal = game.legal_actions()

        # get the current action probabilities from the accumulated regrets
        strategy = self.get_strategy(info, legal)

        # stores the expected payoffs from taking each possible action
        action_payoffs = {}

        for action in legal:
            # copy the game so we can explore each action without changing the original
            next_game = deepcopy(game)
            next_game.step(action)

            # track how likely each player is to reach this point through their own actions
            next_reach = reach.copy()
            next_reach[player] *= strategy[action]

            # explore the rest of the game after taking this action
            action_payoffs[action] = self._traverse(
                next_game,
                next_reach,
                chance,
                regret_updates,
                updating_player
            )

        # calculate the expected payoff of following our current strategy
        # each action's payoff is weighted by how likely we are to choose it
        node_payoff = tuple(
            sum(strategy[a] * action_payoffs[a][p] for a in legal)
            for p in (0, 1)
        )

        # vanilla CFR updates both players, CFR+ can update one player at a time
        if updating_player is None or player == updating_player:

            # create temporary storage for regrets calculated in this iteration
            if info not in regret_updates:
                regret_updates[info] = {a: 0.0 for a in legal}

            for action in legal:
                # regret = payoff if we chose this action - payoff of our current strategy
                # weight by the opponent's reach and probability of the card deal
                regret_updates[info][action] += (
                    chance * reach[1 - player] *
                    (action_payoffs[action][player] - node_payoff[player])
                )

                # record how often we would play each action
                # weight by our own reach and chance -> used for the average strategy
                self.strategy_sum[info][action] += (
                    self._averaging_weight()
                    * chance
                    * reach[player]
                    * strategy[action]
                )

        # return the expected payoffs so earlier decisions can use them
        return node_payoff


    # repeats CFR over every possible card deal to improve the strategy
    def train(self, iterations):
        if iterations <= 0:
            raise ValueError("Iterations must be positive")

        # Kuhn has 3 cards -> 6 possible ways to deal 2 different cards
        deals = list(permutations(["J", "Q", "K"], 2))

        for _ in range(iterations):
            self.iteration += 1

            # store new regrets separately so the strategy doesn't change mid-iteration
            regret_updates = {}

            # explore the full game tree for each of the 6 possible card deals
            for cards in deals:
                game = KuhnPoker()
                game.cards = list(cards)

                # each deal has probability 1/6 and both players start with reach 1
                self._traverse(
                    game,
                    [1.0, 1.0],
                    1 / len(deals),
                    regret_updates
                )

            # once all deals are explored, add the new regrets to the stored regrets
            self._apply_regret_updates(regret_updates)


    def _apply_regret_updates(self, regret_updates):
        # vanilla CFR keeps both positive and negative cumulative regrets
        for info, updates in regret_updates.items():
            for action, delta in updates.items():
                self.regret_sum[info][action] += delta


    # calculates the average strategy learned throughout training
    def average_strategy(self):
        result = {}

        for info, weights in self.strategy_sum.items():
            total = sum(weights.values())

            # if an information set has no accumulated weight, use equal probabilities
            if total == 0:
                result[info] = {
                    action: 1 / len(weights)
                    for action in weights
                }
            else:
                # convert accumulated weights into probabilities that sum to 1
                result[info] = {
                    action: weight / total
                    for action, weight in weights.items()
                }

        return result


    def _averaging_weight(self):
        # vanilla CFR gives every training iteration equal weight
        # CFR+ overrides this to give later iterations more importance
        return 1.0
