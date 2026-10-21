
from copy import deepcopy
from itertools import permutations

import pytest

from poker_ai.games.leduc import LeducPoker


# verify that all sampled deals respect the two copies per rank
def test_card_dealing():
    for seed in range(50):
        game = LeducPoker(seed=seed)
        deal = game.cards + [game._future_public_card]

        assert len(deal) == 3

        for card in ("J", "Q", "K"):
            assert deal.count(card) <= 2


@pytest.mark.parametrize(
    "deal",
    [
        ("J", "J", "J"),
        ("J", "Q", "X"),
        ("J", "Q"),
        ("J", "Q", "K", "K")
    ]
)
def test_invalid_deals(deal):
    with pytest.raises(ValueError):
        LeducPoker(deal=deal)


# ensure the private and future public cards remain hidden
def test_information_sets():
    game = LeducPoker(deal=("J", "Q", "K"))

    assert game.get_information_set() == "J|-|"

    game.step("check")
    assert game.get_information_set() == "Q|-|c"

    game.step("check")
    assert game.get_information_set() == "J|K|cc/"


# check that the opponent card and future public card are not revealed
def test_hidden_information():
    first = LeducPoker(deal=("J", "Q", "K"))
    second = LeducPoker(deal=("J", "K", "Q"))

    assert first.get_information_set() == second.get_information_set()


# check the initial actions in both betting rounds
def test_initial_legal_actions():
    game = LeducPoker(deal=("J", "Q", "K"))

    assert game.legal_actions() == ["check", "bet"]

    game.step("check")
    assert game.legal_actions() == ["check", "bet"]

    game.step("check")
    assert game.legal_actions() == ["check", "bet"]


# after betting the opponent can call, fold or raise
def test_facing_bet():
    game = LeducPoker(deal=("J", "Q", "K"))

    game.step("bet")

    assert game.legal_actions() == ["call", "fold", "raise"]
    assert game.pot == 4
    assert game.contributions == [3, 1]


# a raise must cover the previous bet and add another betting increment
def test_raise_and_call():
    game = LeducPoker(deal=("J", "Q", "K"))

    game.step("bet")
    game.step("raise")

    assert game.contributions == [3, 5]
    assert game.legal_actions() == ["call", "fold"]

    game.step("call")

    assert game.contributions == [5, 5]
    assert game.pot == 10
    assert game.round_index == 1
    assert game.public_card == "K"
    assert game.raises == 0


# verify that the second round uses the larger fixed betting increment
def test_second_round_bets():
    game = LeducPoker(deal=("J", "Q", "K"))

    game.step("check")
    game.step("check")

    game.step("bet")

    assert game.contributions == [5, 1]

    game.step("call")

    assert game.contributions == [5, 5]
    assert game.done
    assert game.get_payoff() == (-5, 5)


# test a fold after an opening bet
def test_fold_payoff():
    game = LeducPoker(deal=("J", "Q", "K"))

    game.step("bet")
    game.step("fold")

    assert game.done
    assert game.get_payoff() == (1, -1)
    assert game.legal_actions() == []


# folding after a raise loses all chips already committed
def test_fold_after_raise():
    game = LeducPoker(deal=("J", "Q", "K"))

    game.step("bet")
    game.step("raise")
    game.step("fold")

    assert game.get_payoff() == (-3, 3)


# a pair must beat a higher unmatched private card
def test_pair_beats_high_card():
    game = LeducPoker(deal=("J", "K", "J"))

    for action in ("check", "check", "check", "check"):
        game.step(action)

    assert game.get_payoff() == (1, -1)


# without a pair the higher private card wins
def test_high_card_showdown():
    game = LeducPoker(deal=("J", "K", "Q"))

    for action in ("check", "check", "check", "check"):
        game.step(action)

    assert game.get_payoff() == (-1, 1)


# equal hand strengths result in a split pot
def test_tied_showdown():
    game = LeducPoker(deal=("Q", "Q", "J"))

    for action in ("check", "check", "check", "check"):
        game.step(action)

    assert game.get_payoff() == (0, 0)


# illegal moves must never change the game state
def test_illegal_actions():
    game = LeducPoker(deal=("J", "Q", "K"))

    with pytest.raises(ValueError):
        game.step("raise")

    assert game.history == ""
    assert game.pot == 2


# recursively check every legal betting sequence for all physical card deals
def test_all_games_are_zero_sum():
    deck = LeducPoker.DECK

    def check_game(game):
        if game.done:
            first, second = game.get_payoff()
            assert first + second == 0
            assert game.pot >= 2
            return

        assert game.legal_actions()

        for action in game.legal_actions():
            next_game = deepcopy(game)
            next_game.step(action)
            check_game(next_game)

    for indices in permutations(range(6), 3):
        deal = tuple(deck[index] for index in indices)
        check_game(LeducPoker(deal=deal))
