
import pytest
from poker_ai.games.kuhn import KuhnPoker
from poker_ai.agents.nash_agent import NashAgent


# TEST 1: Check that our key Nash probabilities are correct
def test_nash_probabilities():
    agent = NashAgent(seed=42)

    # Player 0 bluffs with Jack 1/3 of the time
    assert agent.strategy["J|"]["bet"] == pytest.approx(1/3)

    # Player 0 always bets with King
    assert agent.strategy["K|"]["bet"] == 1

    # Player 1 calls with Queen 1/3 of the time
    assert agent.strategy["Q|b"]["call"] == pytest.approx(1/3)

    # Player 0 calls with Queen 2/3 of the time
    assert agent.strategy["Q|cb"]["call"] == pytest.approx(2/3)

    # Every information set has probabilities summing to 1
    assert len(agent.strategy) == 12

    for probabilities in agent.strategy.values():
        assert sum(probabilities.values()) == pytest.approx(1)


# TEST 2: Check that the agent only chooses legal actions
@pytest.mark.parametrize(
    "info, actions",
    [
        ("J|", ["check", "bet"]),
        ("Q|", ["check", "bet"]),
        ("K|", ["check", "bet"]),
        ("J|c", ["check", "bet"]),
        ("Q|c", ["check", "bet"]),
        ("K|c", ["check", "bet"]),
        ("J|b", ["call", "fold"]),
        ("Q|b", ["call", "fold"]),
        ("K|b", ["call", "fold"]),
        ("J|cb", ["call", "fold"]),
        ("Q|cb", ["call", "fold"]),
        ("K|cb", ["call", "fold"]),
    ],
)
def test_nash_legal_actions(info, actions):
    agent = NashAgent(seed=42)

    # Every chosen action must belong to the legal actions
    for _ in range(20):
        assert agent.choose_action(info, actions) in actions


# TEST 3: Check the agent's actual bluffing frequency
def test_nash_bluff_frequency():
    agent = NashAgent(seed=42)

    # Ask the agent to act with Jack 6000 times
    actions = [
        agent.choose_action("J|", ["check", "bet"])
        for _ in range(6000)
    ]

    # Estimate the proportion of times it bets
    bluff_rate = actions.count("bet") / len(actions)

    # Should be reasonably close to 1/3
    assert abs(bluff_rate - 1/3) < 0.03


# TEST 4: Check that the agent can play a complete hand
def test_nash_complete_game():
    game = KuhnPoker()
    agent = NashAgent(seed=42)

    # Both seats use the equilibrium agent
    while not game.done:
        info = game.get_information_set()
        actions = game.legal_actions()

        chosen = agent.choose_action(info, actions)
        game.step(chosen)

    # Game should finish with valid zero-sum payouts
    assert game.done is True
    assert sum(game.get_payoff()) == 0


# TEST 5: Reject unknown information sets
def test_invalid_information_set():
    agent = NashAgent()

    with pytest.raises(ValueError):
        agent.choose_action("X|", ["check", "bet"])


from poker_ai.experiments.simulate import simulate


# Check that the simulation returns a valid average payoff
def test_simulation():
    result = simulate(n_hands=100, seed=42)

    # Player 0's average payoff must be between -2 and +2
    assert -2 <= result <= 2


# Check that the same seed produces identical results
def test_simulation_reproducibility():
    result_1 = simulate(n_hands=1000, seed=42)
    result_2 = simulate(n_hands=1000, seed=42)

    assert result_1 == result_2
