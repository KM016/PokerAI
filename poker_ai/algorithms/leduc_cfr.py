from poker_ai.algorithms.cfr import CFRTrainer
from poker_ai.algorithms.cfr_plus import CFRPlusTrainer
from poker_ai.evaluation.leduc_evaluate import possible_deals
from poker_ai.games.leduc import LeducPoker


# trains leduc poker using the existing vanilla cfr algorithm
class LeducCFRTrainer(CFRTrainer):

    def train(self, iterations):
        # check that the requested number of training iterations is valid
        if iterations <= 0:
            raise ValueError("Iterations must be positive")

        # get all distinct leduc card deals and their probabilities
        # identical physical cards are combined into the same rank deal
        deals = possible_deals()

        # each iteration explores the entire leduc game tree
        for _ in range(iterations):
            self.iteration += 1

            # store regrets separately so strategies stay fixed during the iteration
            regret_updates = {}

            # calculate regrets across every possible private and public card deal
            for deal, probability in deals:
                game = LeducPoker(deal=deal)

                # reuse the same cfr traversal we implemented for kuhn poker
                # the traversal calculates action values and counterfactual regrets
                self._traverse(
                    game,
                    [1.0, 1.0],
                    probability,
                    regret_updates
                )

            # update cumulative regrets after exploring all card deals
            self._apply_regret_updates(regret_updates)


# trains leduc poker using the existing cfr+ algorithm
class LeducCFRPlusTrainer(CFRPlusTrainer):

    def train(self, iterations):
        # check that the requested number of training iterations is valid
        if iterations <= 0:
            raise ValueError("Iterations must be positive")

        # reuse the same leduc deals and chance probabilities as vanilla cfr
        deals = possible_deals()

        for _ in range(iterations):
            self.iteration += 1

            # cfr+ updates the players separately rather than simultaneously
            # player 1 can respond to the updated strategy of player 0
            for player in (0, 1):
                regret_updates = {}

                # explore every possible deal for the player being updated
                for deal, probability in deals:
                    game = LeducPoker(deal=deal)

                    # use the inherited traversal to calculate this players regrets
                    self._traverse(
                        game,
                        [1.0, 1.0],
                        probability,
                        regret_updates,
                        updating_player=player
                    )

                # clip negative regrets to zero using our existing cfr+ method
                self._apply_regret_updates(regret_updates)
