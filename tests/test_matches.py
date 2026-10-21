
import pytest

from poker_ai.evaluation.matches import play_match, balanced_exact_ev
from poker_ai.agents.nash_agent import NashAgent


# Simple opponent that always checks or folds
class PassiveAgent:
    def choose_action(self, info_set, legal_actions):
        return "check" if "check" in legal_actions else "fold"


# TEST 1: Verify match statistics
def test_match_counts():
    result = play_match(
        NashAgent(seed=1),
        NashAgent(seed=2),
        n_hands=20,
        seed=3
    )

    assert result["hands"] == 20
    assert result["wins_a"] + result["losses_a"] == 20

    assert result["avg_payoff_a"] == pytest.approx(
        result["total_payoff_a"] / 20
    )


# TEST 2: Identical deterministic agents should balance out
def test_identical_passive_agents_cancel_over_paired_deals():
    result = play_match(
        PassiveAgent(),
        PassiveAgent(),
        n_hands=40,
        seed=3
    )

    assert result["total_payoff_a"] == 0


# TEST 3: Same seeds should produce the same results
def test_match_reproducible_with_seeded_agents():
    first = play_match(
        NashAgent(seed=1),
        NashAgent(seed=2),
        n_hands=100,
        seed=3
    )

    second = play_match(
        NashAgent(seed=1),
        NashAgent(seed=2),
        n_hands=100,
        seed=3
    )

    assert first == second


# TEST 4: Reject invalid hand counts
@pytest.mark.parametrize("n_hands", [0, -2, 3])
def test_reject_invalid_hand_counts(n_hands):
    with pytest.raises(ValueError):
        play_match(
            PassiveAgent(),
            PassiveAgent(),
            n_hands=n_hands
        )


def test_balanced_exact_ev():
    strategy = NashAgent().strategy

    # Identical strategies playing equally often in both
    # positions must have zero average payoff against each other
    result = balanced_exact_ev(strategy, strategy)

    assert result == pytest.approx(0.0)
