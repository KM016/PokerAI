from poker_ai.agents.strategy_agent import StrategyAgent
from poker_ai.evaluation.evaluate import best_response

# Agent that plays the mathematically best strategy againts the opponent

class BestResponseAgent(StrategyAgent):
    def __init__(self, opponent_strategy, seed=None):
        # calculate the best response for each player position
        _, strategy_0 = best_response(opponent_strategy, player=0)
        _, strategy_1 = best_response(opponent_strategy, player=1)

        # player 0 and player 1 have different information sets, so we can combine their strategies into one dictionary
        combined_strategy = {
            **strategy_0,
            **strategy_1
        }

        # reuse StrategyAgent to select actions during games
        super().__init__(combined_strategy, seed=seed)
