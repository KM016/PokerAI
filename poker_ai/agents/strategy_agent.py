import random

# take a strategy we have already calculated and use it to make decisions in a game 

class StrategyAgent:
    def __init__(self, strategy, seed=None):
        # store a copy of the learned strategy
        self.strategy = {
            info: actions.copy()
            for info, actions in strategy.items()
        }

        # local RNG for reproducible decisions
        self.rng = random.Random(seed)

    def choose_action(self, info_set, legal_actions):
        # check that this information set is recognised
        if info_set not in self.strategy:
            raise ValueError(f"Unknown information set: {info_set}")

        probabilities = self.strategy[info_set]

        # check that the learned strategy matches the legal actions
        if set(probabilities) != set(legal_actions):
            raise ValueError("Strategy actions do not match legal actions")

        # get the probability of every legal action
        weights = [
            probabilities[action]
            for action in legal_actions
        ]

        # validate the probability distribution
        if (
            any(prob < 0 for prob in weights)
            or not 0.999999 <= sum(weights) <= 1.000001
        ):
            raise ValueError("Invalid strategy probabilities")

        # randomly choose an action using the learned probabilities
        return self.rng.choices(
            legal_actions,
            weights=weights,
            k=1
        )[0]
