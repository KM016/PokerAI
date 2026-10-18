
import pytest
from poker_ai.games.kuhn import KuhnPoker


# TEST 1: Check that a newly created game starts correctly
def test_initial_state():
    game = KuhnPoker()

    # Check that exactly 2 cards are dealt
    assert len(game.cards) == 2

    # Converting to a set removes duplicates, so both cards must be different
    assert len(set(game.cards)) == 2

    # Check that both cards belong to the Kuhn deck (J, Q, K)
    assert set(game.cards).issubset({"J", "Q", "K"})

    # Player 0 should always act first
    assert game.current_player == 0

    # No actions should have been taken yet
    assert game.history == ""

    # The game should not be finished
    assert game.done is False


# TEST 2: Check which actions are legal for different betting histories

# parametrize runs the same test multiple times using different inputs
# Each tuple contains a betting history and its expected legal actions
@pytest.mark.parametrize(
    "history, expected",
    [
        ("", ["check", "bet"]),    # No actions yet
        ("c", ["check", "bet"]),   # Player 0 checked
        ("b", ["call", "fold"]),   # Player 0 bet
        ("cb", ["call", "fold"]),  # Player 0 checked, Player 1 bet
    ],
)
def test_legal_actions(history, expected):
    game = KuhnPoker()

    # Manually set the history to the one being tested
    game.history = history

    # Check that the available actions match what we expect
    assert game.legal_actions() == expected


# TEST 3: Check that a finished game has no legal actions
def test_finished_game():
    game = KuhnPoker()

    # Manually mark the game as finished
    game.done = True

    # An empty list means no further actions are allowed
    assert game.legal_actions() == []


# TEST 4: Check that making moves correctly updates the game state
def test_step_updates_state():
    game = KuhnPoker()

    # Player 0 checks
    game.step("check")

    # The check should be recorded as "c"
    assert game.history == "c"

    # It should now be Player 1's turn
    assert game.current_player == 1

    # The game should still be ongoing
    assert game.done is False

    # Player 1 bets
    game.step("bet")

    # History should now show check followed by bet
    assert game.history == "cb"

    # It should now be Player 0's turn to respond
    assert game.current_player == 0

    # The game should still be ongoing
    assert game.done is False


# TEST 5: Check that every possible ending correctly finishes the game

# Test all 5 possible terminal betting sequences
# Each tuple contains the actions taken and the expected final history
@pytest.mark.parametrize(
    "actions, expected_history",
    [
        (["check", "check"], "cc"),       # Both check
        (["bet", "call"], "bk"),          # Player 0 bets, Player 1 calls
        (["bet", "fold"], "bf"),          # Player 0 bets, Player 1 folds
        (["check", "bet", "call"], "cbk"), # Player 0 checks, Player 1 bets, Player 0 calls
        (["check", "bet", "fold"], "cbf"), # Player 0 checks, Player 1 bets, Player 0 folds
    ],
)
def test_step_ends_game(actions, expected_history):
    game = KuhnPoker()

    # Execute every action in the chosen sequence
    for action in actions:
        game.step(action)

    # Check that the complete betting history was recorded correctly
    assert game.history == expected_history

    # Check that the game is marked as finished
    assert game.done is True

    # No further actions should be available
    assert game.legal_actions() == []


# TEST 6: Check that illegal actions are rejected
def test_invalid_action():
    game = KuhnPoker()

    # Calling is illegal at the start because nobody has bet
    # pytest.raises checks that step() produces a ValueError
    with pytest.raises(ValueError):
        game.step("call")

    # The rejected action must not change the betting history
    assert game.history == ""

    # It should still be Player 0's turn
    assert game.current_player == 0


# TEST 7: Check that all possible endings produce correct payouts
@pytest.mark.parametrize(
    "cards, actions, expected",
    [
        # Both check: higher card wins 1 chip
        (["K", "J"], ["check", "check"], (1, -1)),
        (["J", "K"], ["check", "check"], (-1, 1)),

        # Bet and call: higher card wins 2 chips
        (["K", "Q"], ["bet", "call"], (2, -2)),
        (["Q", "K"], ["bet", "call"], (-2, 2)),

        # Check, bet, call: higher card wins 2 chips
        (["K", "J"], ["check", "bet", "call"], (2, -2)),
        (["J", "K"], ["check", "bet", "call"], (-2, 2)),

        # Player 1 folds: Player 0 wins regardless of cards
        (["J", "K"], ["bet", "fold"], (1, -1)),

        # Player 0 folds: Player 1 wins regardless of cards
        (["K", "J"], ["check", "bet", "fold"], (-1, 1)),
    ],
)
def test_get_payoff(cards, actions, expected):
    game = KuhnPoker()

    # Set specific cards so we know who should win
    game.cards = cards

    # Play all actions until the hand finishes
    for action in actions:
        game.step(action)

    # Check that both players receive the correct payouts
    assert game.get_payoff() == expected

    # Total profit must always equal zero
    assert sum(game.get_payoff()) == 0


# TEST 8: Payouts cannot be calculated before the hand ends
def test_payoff_before_game_ends():
    game = KuhnPoker()

    # Expect an error because the game is still ongoing
    with pytest.raises(ValueError):
        game.get_payoff()


# TEST 9: Check that players receive the correct information set
@pytest.mark.parametrize(
    "actions, expected",
    [
        ([], "J|"),                     # Player 0 hasn't acted
        (["check"], "Q|c"),             # Player 1 acts after a check
        (["bet"], "Q|b"),               # Player 1 responds to a bet
        (["check", "bet"], "J|cb"),      # Player 0 responds to a bet
    ],
)
def test_information_set(actions, expected):
    game = KuhnPoker()

    # Fix the cards so our test results are predictable
    game.cards = ["J", "Q"]

    # Play the actions in the test case
    for action in actions:
        game.step(action)

    # Check that the current player sees the correct information
    assert game.get_information_set() == expected


# TEST 10: Check that the opponent's card is hidden
def test_information_set_hides_opponent_card():
    game = KuhnPoker()

    # Player 0 has Jack, opponent has Queen
    game.cards = ["J", "Q"]
    first_info = game.get_information_set()

    # Change only the opponent's card to King
    game.cards = ["J", "K"]
    second_info = game.get_information_set()

    # Player 0's information must remain identical
    assert first_info == second_info == "J|"


# TEST 11: Check that finished games reject information requests
def test_information_set_finished_game():
    game = KuhnPoker()

    # Complete the hand
    game.step("check")
    game.step("check")

    # No further decision is possible
    with pytest.raises(ValueError):
        game.get_information_set()
