
import pytest

from poker_ai.agents.nash_agent import NashAgent
from poker_ai.agents.strategy_agent import StrategyAgent
from poker_ai.agents.strategy_io import save_strategy
from poker_ai.cli import make_opponent, play_hand, prompt_action
from poker_ai.games.kuhn import KuhnPoker


# check that actions can be selected using numbers or names
@pytest.mark.parametrize(
    "choice, expected",
    [
        ("1", "check"),
        ("check", "check"),
        ("2", "bet"),
        ("BET", "bet")
    ]
)
def test_prompt_action_accepts_numbers_and_names(monkeypatch, choice, expected):
    monkeypatch.setattr("builtins.input", lambda _: choice)

    assert prompt_action(["check", "bet"]) == expected


# invalid inputs should be rejected until a legal action is entered
def test_prompt_action_retries_invalid_choices(monkeypatch, capsys):
    choices = iter(["0", "3", "1"])
    monkeypatch.setattr("builtins.input", lambda _: next(choices))

    assert prompt_action(["call", "fold"]) == "call"
    assert "Invalid action" in capsys.readouterr().out


# check that an opponent can be selected using its name
def test_make_opponent_accepts_nash_name():
    assert isinstance(make_opponent("Nash"), NashAgent)


# check that the CLI can load a previously saved CFR+ strategy
def test_make_opponent_loads_saved_cfr_plus(monkeypatch, tmp_path):
    strategy = NashAgent().strategy
    save_strategy(strategy, tmp_path / "models/kuhn/cfr_plus.json")
    monkeypatch.chdir(tmp_path)

    agent = make_opponent("CFR+")

    assert isinstance(agent, StrategyAgent)
    assert agent.strategy == strategy


# play a complete hand with fixed cards and a passive opponent
def test_play_hand_with_numbered_action(monkeypatch, capsys):
    class FixedGame(KuhnPoker):
        def __init__(self):
            super().__init__()
            self.cards = ["K", "J"]

    class PassiveAgent:
        def choose_action(self, info_set, legal_actions):
            return "check" if "check" in legal_actions else "fold"

    monkeypatch.setattr("poker_ai.cli.KuhnPoker", FixedGame)
    monkeypatch.setattr("builtins.input", lambda _: "1")

    assert play_hand(PassiveAgent(), human_player=0) == 1

    output = capsys.readouterr().out
    assert "Your card: K" in output
    assert "AI chooses: check" in output
    assert "Hand payoff: +1 chips" in output
