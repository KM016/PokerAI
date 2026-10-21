import json

import pytest

from poker_ai.agents.strategy_agent import StrategyAgent
from poker_ai.agents.strategy_io import load_strategy
from poker_ai.algorithms.q_learning import QLearningAgent
from poker_ai.experiments.agent_tournament import baseline_strategy
from poker_ai.experiments.train_q_learning import (
    greedy_strategy, learning_rates, play_training_hand, run_experiment
)


class FixedDeal:
    def sample(self, cards, count):
        return ["K", "J"]


def test_terminal_reward_updates_q_value():
    learner = QLearningAgent(alpha=1, epsilon=0, seed=1)
    learner.q_table["K|"] = {"check": -1.0, "bet": 0.0}
    opponent = StrategyAgent(baseline_strategy(), seed=2)

    payoff = play_training_hand(learner, opponent, 0, FixedDeal())

    assert payoff == 1
    assert learner.q_table["K|"]["bet"] == 1.0


def test_next_decision_is_used_before_terminal_payoff():
    learner = QLearningAgent(alpha=1, epsilon=0, seed=1)
    learner.q_table["J|"] = {"check": 0.0, "bet": -2.0}
    learner.q_table["J|cb"] = {"call": -2.0, "fold": -1.0}

    class BettingOpponent:
        def choose_action(self, info_set, legal_actions):
            return "bet" if "bet" in legal_actions else "fold"

    class JackDeal:
        def sample(self, cards, count):
            return ["J", "K"]

    payoff = play_training_hand(learner, BettingOpponent(), 0, JackDeal())

    assert payoff == -1
    assert learner.q_table["J|"]["check"] == -1.0
    assert learner.q_table["J|cb"]["fold"] == -1.0


def test_greedy_strategy_is_complete_and_does_not_change_q_table():
    learner = QLearningAgent()
    learner.q_table["K|"] = {"check": 0.0, "bet": 1.0}

    strategy = greedy_strategy(learner)

    assert len(strategy) == 12
    assert strategy["K|"] == {"check": 0.0, "bet": 1.0}
    assert strategy["J|b"] == {"call": 0.5, "fold": 0.5}
    assert "J|b" not in learner.q_table
    assert all(abs(sum(probs.values()) - 1) < 1e-12 for probs in strategy.values())


def test_training_creates_reusable_model_and_csv(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    results = run_experiment(checkpoints=(2, 4), opponent_name="Passive", schedule="Constant", seed=10)

    assert len(results) == 2
    assert results[0]["hands"] == 2
    assert results[1]["hands"] == 4
    assert (tmp_path / "models/kuhn/q_learning_passive_constant.json").exists()
    assert (tmp_path / "results/kuhn/q_learning_passive_constant.csv").exists()


# use the same decaying alpha values as the earlier stability experiment
def test_decay_alpha_matches_stability_schedule():
    assert learning_rates("Decay alpha", 0, 100) == pytest.approx((0.1, 0.1))
    assert learning_rates("Decay alpha", 99, 100) == pytest.approx((0.01, 0.1))
    assert learning_rates("Decay both", 99, 100) == pytest.approx((0.01, 0.01))


# check that the tuned run saves all three files and the correct settings
def test_tuned_training_saves_models_and_metadata(tmp_path):
    results = run_experiment(
        checkpoints=(2, 20), seed=101,
        opponent_name="Random", schedule="Decay alpha",
        model_dir=tmp_path / "models", output_dir=tmp_path / "results"
    )
    prefix = "q_learning_random_decay_alpha"
    policy = load_strategy(tmp_path / "models" / f"{prefix}.json")
    q_values = load_strategy(tmp_path / "models" / f"{prefix}_q_values.json")
    with (tmp_path / "models" / f"{prefix}_metadata.json").open() as file:
        metadata = json.load(file)

    assert len(policy) == 12
    assert q_values
    assert metadata["training_hands"] == 20
    assert metadata["seed"] == 101
    assert metadata["schedule"] == "Decay alpha"
    assert metadata["final_alpha"] == pytest.approx(0.01)
    assert metadata["final_epsilon"] == 0.1
    assert metadata["evaluation"] == results[-1]
    assert (tmp_path / "results" / f"{prefix}.csv").exists()


# make sure mixed training also works with the same training function
def test_mixed_training_experiences_opponents(tmp_path):
    results = run_experiment(
        checkpoints=(2, 12), opponent_name="Mixed", schedule="Decay epsilon",
        seed=13, model_dir=tmp_path / "models", output_dir=tmp_path / "results"
    )
    assert len(results) == 2
    assert 0 < results[-1]["visited_info_sets"] <= 12


# repeated training with the same seed must produce the same saved strategy
def test_training_is_reproducible(tmp_path):
    kwargs = dict(checkpoints=(2, 20), seed=101, opponent_name="Random", schedule="Decay alpha")
    first = run_experiment(**kwargs, model_dir=tmp_path / "a", output_dir=tmp_path / "out_a")
    second = run_experiment(**kwargs, model_dir=tmp_path / "b", output_dir=tmp_path / "out_b")
    assert first == second
    prefix = "q_learning_random_decay_alpha.json"
    assert load_strategy(tmp_path / "a" / prefix) == load_strategy(tmp_path / "b" / prefix)


@pytest.mark.parametrize("checkpoints", [(0,), (), (2, 1), (1, 3)])
def test_invalid_checkpoints_raise(checkpoints):
    with pytest.raises(ValueError, match="Checkpoints|Total hands"):
        run_experiment(checkpoints=checkpoints)


@pytest.mark.parametrize("opponent_name,schedule", [
    ("Unknown", "Constant"), ("Random", "Unknown")
])
def test_invalid_configuration_raises(opponent_name, schedule):
    with pytest.raises(ValueError, match="Unknown"):
        run_experiment(checkpoints=(2, 4), opponent_name=opponent_name, schedule=schedule)
