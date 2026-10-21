
import pytest

from poker_ai.evaluation.leduc_evaluate import (
    possible_deals,
    information_sets,
    baseline_strategy,
    exact_ev,
    balanced_exact_ev,
    best_response,
    exploitability,
)


# physical card combinations reduce to 24 distinct rank deals
# their probabilities must still represent all 120 physical deals
def test_deal_probabilities():
    deals = possible_deals()

    assert len(deals) == 24
    assert sum(probability for _, probability in deals) == pytest.approx(1)


# verify the complete information set structure
def test_information_sets():
    infos = information_sets()

    assert len(infos) == 288
    assert len({info for info, _, _ in infos}) == 288


# make sure our baseline opponents define valid action probabilities
@pytest.mark.parametrize("style", ["Passive", "Random", "Rule-based"])
def test_baseline_strategies(style):
    strategy = baseline_strategy(style)

    assert len(strategy) == 288

    for actions in strategy.values():
        assert sum(actions.values()) == pytest.approx(1)
        assert all(probability >= 0 for probability in actions.values())


# two passive players always check so neither has a seat advantage
def test_passive_against_passive():
    passive = baseline_strategy("Passive")

    first, second = exact_ev(passive, passive)

    assert first == pytest.approx(0)
    assert second == pytest.approx(0)


# all exact payoffs must add up to zero
def test_exact_ev_is_zero_sum():
    passive = baseline_strategy("Passive")
    random = baseline_strategy("Random")

    first, second = exact_ev(passive, random)

    assert first + second == pytest.approx(0)


# playing the same strategy in both positions has zero balanced payoff
def test_balanced_self_play():
    random = baseline_strategy("Random")

    assert balanced_exact_ev(random, random) == pytest.approx(0)


# an optimal best response must do at least as well as a passive response
@pytest.mark.parametrize("player", [0, 1])
def test_best_response_improves_payoff(player):
    opponent = baseline_strategy("Random")
    passive = baseline_strategy("Passive")

    best_value, response = best_response(opponent, player)

    if player == 0:
        reference = exact_ev(passive, opponent)[0]
        calculated = exact_ev(response, opponent)[0]
    else:
        reference = exact_ev(opponent, passive)[1]
        calculated = exact_ev(opponent, response)[1]

    assert best_value >= reference - 1e-9
    assert best_value == pytest.approx(calculated)


# the best response must choose a legal action at every information set
@pytest.mark.parametrize("player", [0, 1])
def test_best_response_strategy_is_valid(player):
    opponent = baseline_strategy("Random")
    _, response = best_response(opponent, player)

    infos = {
        info: actions
        for info, acting_player, actions in information_sets()
        if acting_player == player
    }

    assert set(response) == set(infos)

    for info, actions in infos.items():
        assert set(response[info]) == set(actions)
        assert sum(response[info].values()) == pytest.approx(1)


# passive and random strategies should both be exploitable
def test_exploitability():
    passive = baseline_strategy("Passive")
    random = baseline_strategy("Random")

    assert exploitability(passive) > 0
    assert exploitability(random) > 0


# unknown baseline names should be rejected
def test_invalid_baseline():
    with pytest.raises(ValueError):
        baseline_strategy("Unknown")


# missing probability distributions must not be accepted
def test_invalid_strategy():
    passive = baseline_strategy("Passive")
    passive.pop("J|-|")

    with pytest.raises(ValueError):
        exact_ev(passive, passive)
