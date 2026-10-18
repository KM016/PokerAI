
import pytest

from poker_ai.agents.nash_agent import NashAgent
from poker_ai.evaluation.evaluate import exact_ev, best_response, exploitability

# TEST 1: Exact Nash EV should match the theoretical result
def test_exact_nash_ev():
    agent = NashAgent()

    ev_0, ev_1 = exact_ev(agent.strategy, agent.strategy)

    # Player 0's theoretical EV is -1/18
    assert ev_0 == pytest.approx(-1/18)

    # Player 1 receives the opposite payoff
    assert ev_1 == pytest.approx(1/18)


# TEST 2: The game must remain zero-sum
def test_exact_ev_zero_sum():
    agent = NashAgent()

    ev_0, ev_1 = exact_ev(agent.strategy, agent.strategy)

    # Both expected profits must sum to zero
    assert ev_0 + ev_1 == pytest.approx(0)


# TEST 3: Exact calculations should always give the same result
def test_exact_ev_reproducibility():
    agent = NashAgent()

    result_1 = exact_ev(agent.strategy, agent.strategy)
    result_2 = exact_ev(agent.strategy, agent.strategy)

    # Unlike simulation, no random variation should occur
    assert result_1 == result_2


# TEST 4: Nash should have the correct best-response values
def test_best_response_nash():
    agent = NashAgent()

    # Find the strongest possible response against Nash
    br0_ev, _ = best_response(agent.strategy, player=0)
    br1_ev, _ = best_response(agent.strategy, player=1)

    # Neither player can improve beyond the game's equilibrium value
    assert br0_ev == pytest.approx(-1/18)
    assert br1_ev == pytest.approx(1/18)


# TEST 5: Nash equilibrium should have zero exploitability
def test_nash_exploitability():
    agent = NashAgent()

    assert exploitability(agent.strategy) == pytest.approx(0, abs=1e-12)


# TEST 6: A weak strategy should be exploitable
def test_weak_strategy_exploitability():
    agent = NashAgent()

    # Construct an agent that always checks or folds
    weak_strategy = {}

    for info, probabilities in agent.strategy.items():
        legal = list(probabilities)
        chosen = "check" if "check" in legal else "fold"

        weak_strategy[info] = {
            action: float(action == chosen)
            for action in legal
        }

    # An optimal opponent can win 1 chip per hand in either seat
    assert best_response(weak_strategy, 0)[0] == pytest.approx(1)
    assert best_response(weak_strategy, 1)[0] == pytest.approx(1)

    # Average exploitability should equal 1
    assert exploitability(weak_strategy) == pytest.approx(1)


# TEST 7: Reject invalid player numbers
def test_invalid_best_response_player():
    agent = NashAgent()

    with pytest.raises(ValueError):
        best_response(agent.strategy, player=2)

