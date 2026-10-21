
import pytest

from poker_ai.agents.strategy_agent import StrategyAgent
from poker_ai.algorithms.cfr_plus import CFRPlusTrainer
from poker_ai.games.kuhn import KuhnPoker


def test_strategy_agent_selects_action():
    agent = StrategyAgent({
        "J|": {"check": 1.0, "bet": 0.0}
    })

    assert agent.choose_action("J|", ["check", "bet"]) == "check"


def test_strategy_agent_reproducibility():
    strategy = {
        "J|": {"check": 0.4, "bet": 0.6}
    }

    agent1 = StrategyAgent(strategy, seed=42)
    agent2 = StrategyAgent(strategy, seed=42)

    actions1 = [
        agent1.choose_action("J|", ["check", "bet"])
        for _ in range(100)
    ]

    actions2 = [
        agent2.choose_action("J|", ["check", "bet"])
        for _ in range(100)
    ]

    assert actions1 == actions2


def test_strategy_agent_invalid_info():
    agent = StrategyAgent({
        "J|": {"check": 0.5, "bet": 0.5}
    })

    with pytest.raises(ValueError):
        agent.choose_action("K|", ["check", "bet"])


def test_trained_agent_plays_complete_hand():
    # Train our CFR+ algorithm
    trainer = CFRPlusTrainer()
    trainer.train(100)

    # Create two agents using the learned strategy
    agents = [
        StrategyAgent(trainer.average_strategy(), seed=42),
        StrategyAgent(trainer.average_strategy(), seed=43)
    ]

    game = KuhnPoker()
    game.reset()

    # Let both agents play until the hand finishes
    while not game.done:
        player = game.current_player
        info = game.get_information_set()
        legal = game.legal_actions()

        action = agents[player].choose_action(info, legal)
        game.step(action)

    # Kuhn Poker is a zero-sum game
    payoff0, payoff1 = game.get_payoff()

    assert payoff0 + payoff1 == 0
