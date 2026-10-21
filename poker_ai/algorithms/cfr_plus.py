from poker_ai.algorithms.cfr import CFRTrainer
from itertools import permutations
from poker_ai.games.kuhn import KuhnPoker


# trains a poker strategy using CFR+ -> an improved version of vanilla CFR
class CFRPlusTrainer(CFRTrainer):

    def __init__(self, averaging_delay=0):
        # use CFRs existing regret and strategy storage
        super().__init__()

        # averaging delay determines how many early iterations we ignore
        if averaging_delay < 0:
            raise ValueError("Averaging delay cannot be negative")

        self.averaging_delay = averaging_delay

    def _averaging_weight(self):
        # give later iterations more weight as their strategies are generally better
        # early iterations within the delay receive a weight of 0
        return max(0, self.iteration - self.averaging_delay)

    def _apply_regret_updates(self, regret_updates):
        # unlike vanilla CFR, CFR+ does not keep negative cumulative regrets
        for info, updates in regret_updates.items():
            for action, delta in updates.items():
                # add the new regret but clip the result at 0 if negative
                self.regret_sum[info][action] = max(
                    0.0,
                    self.regret_sum[info][action] + delta
                )

    # trains CFR+ by updating each players regrets separately
    def train(self, iterations):
        if iterations <= 0:
            raise ValueError("Iterations must be positive")

        # Kuhn has 6 possible card deals -> each has probability 1/6
        deals = list(permutations(["J", "Q", "K"], 2))

        for _ in range(iterations):
            self.iteration += 1

            # update player 0 first, then player 1
            # player 1 can then respond to player 0s newly updated strategy
            for player in (0, 1):
                regret_updates = {}

                # explore every possible deal to calculate this players regrets
                for cards in deals:
                    game = KuhnPoker()
                    game.cards = list(cards)

                    # explore the game tree but only record regrets for this player
                    self._traverse(
                        game,
                        [1.0, 1.0],
                        1 / len(deals),
                        regret_updates,
                        updating_player=player
                    )

                # update this player's cumulative regrets before moving to the next player
                self._apply_regret_updates(regret_updates)
