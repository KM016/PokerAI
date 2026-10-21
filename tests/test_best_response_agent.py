
import pytest

from poker_ai.agents.best_response_agent import BestResponseAgent
from poker_ai.evaluation.matches import balanced_exact_ev
from poker_ai.experiments.agent_tournament import baseline_strategy


def test_best_response_agent_has_complete_strategy():
    passive = baseline_strategy()
    agent = BestResponseAgent(passive)

    # Six information sets per player
    assert len(agent.strategy) == 12

    # The best response selects a deterministic action
    for actions in agent.strategy.values():
        assert sum(actions.values()) == pytest.approx(1.0)
        assert all(p in (0.0, 1.0) for p in actions.values())


def test_best_response_exploits_passive_agent():
    passive = baseline_strategy()
    agent = BestResponseAgent(passive)

    # An optimal opponent earns one unit per hand
    ev = balanced_exact_ev(agent.strategy, passive)

    assert ev == pytest.approx(1.0)
