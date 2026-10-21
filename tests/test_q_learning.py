
import pytest

from poker_ai.algorithms.q_learning import QLearningAgent


def test_q_values_start_at_zero():
    agent = QLearningAgent(seed=1)
    agent.choose_action("J|", ["check", "bet"])

    assert agent.q_table["J|"] == {"check": 0.0, "bet": 0.0}


def test_greedy_action():
    agent = QLearningAgent(epsilon=0, seed=1)
    agent.q_table["K|"] = {"check": 0.5, "bet": 1.0}

    assert agent.choose_action("K|", ["check", "bet"]) == "bet"


def test_epsilon_explores():
    agent = QLearningAgent(epsilon=1, seed=1)
    agent.q_table["K|"] = {"check": 0.5, "bet": 1.0}

    choices = {
        agent.choose_action("K|", ["check", "bet"])
        for _ in range(40)
    }

    assert choices == {"check", "bet"}


def test_terminal_update():
    agent = QLearningAgent(alpha=0.5)

    agent.update("J|b", "fold", -1, ["call", "fold"])

    assert agent.q_table["J|b"]["fold"] == -0.5


def test_bootstrap_from_next_agent_decision():
    agent = QLearningAgent(alpha=0.5, gamma=1)
    agent.q_table["J|cb"] = {"call": 2.0, "fold": 0.0}

    agent.update(
        "J|",
        "check",
        0,
        ["check", "bet"],
        "J|cb",
        ["call", "fold"]
    )

    assert agent.q_table["J|"]["check"] == 1.0


def test_reproducible_choices():
    a = QLearningAgent(epsilon=0.3, seed=21)
    b = QLearningAgent(epsilon=0.3, seed=21)

    assert [
        a.choose_action("Q|", ["check", "bet"])
        for _ in range(40)
    ] == [
        b.choose_action("Q|", ["check", "bet"])
        for _ in range(40)
    ]


def test_mismatched_legal_actions_raise():
    agent = QLearningAgent()
    agent.choose_action("J|", ["check", "bet"])

    with pytest.raises(ValueError, match="Q-table actions"):
        agent.choose_action("J|", ["call", "fold"])
