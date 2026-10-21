import csv
import json
import random

import pytest
import torch

from poker_ai.agents.dqn_agent import DQNAgent
from poker_ai.agents.strategy_io import load_strategy
from poker_ai.algorithms.dqn import DQNNetwork
from poker_ai.experiments.train_dqn import (
    greedy_strategy, play_training_hand, run_experiment
)


class ScriptedLearner:
    def __init__(self):
        self.experiences = []

    def choose_action(self, info_set, legal_actions):
        return "check" if "check" in legal_actions else "fold"

    def remember(self, *args):
        self.experiences.append(args)


class AlwaysBetOpponent:
    def choose_action(self, info_set, legal_actions):
        return "bet" if "bet" in legal_actions else "call"


# learner check, opponent bet, learner fold must create a two step transition
def test_training_hand_stores_transition_between_learner_decisions():
    learner = ScriptedLearner()
    payoff = play_training_hand(learner, AlwaysBetOpponent(), 0, random.Random(9))

    assert payoff == -1
    assert len(learner.experiences) == 2
    start, finish = learner.experiences
    assert start[0].endswith("|")
    assert start[1] == "check"
    assert start[2] == 0
    assert start[4] == start[0] + "cb"
    assert start[5] == ["call", "fold"]
    assert finish[0] == start[4]
    assert finish[1] == "fold"
    assert finish[2] == -1
    assert len(finish) == 4


# a player in seat one must receive the payoff from player one, not player zero
def test_training_hand_uses_correct_seat_reward():
    learner = ScriptedLearner()
    payoff = play_training_hand(learner, AlwaysBetOpponent(), 1, random.Random(9))

    assert payoff == -1
    assert len(learner.experiences) == 1
    assert learner.experiences[0][1] == "fold"
    assert learner.experiences[0][2] == -1


def test_training_hand_rejects_invalid_seat():
    with pytest.raises(ValueError):
        play_training_hand(ScriptedLearner(), AlwaysBetOpponent(), 2, random.Random(9))


# evaluation must use predicted q values without epsilon random actions
def test_greedy_strategy_uses_correct_action_names():
    learner = DQNAgent(seed=10, epsilon=1)
    with torch.no_grad():
        for parameter in learner.policy_net.parameters():
            parameter.zero_()
        learner.policy_net.network[-1].bias.copy_(torch.tensor([2.0, 1.0]))

    strategy = greedy_strategy(learner)
    assert len(strategy) == 12
    assert strategy["J|"] == {"check": 1.0, "bet": 0.0}
    assert strategy["K|cb"] == {"call": 1.0, "fold": 0.0}
    assert all(sum(actions.values()) == 1.0 for actions in strategy.values())


# save both json policy and pt weights, with metrics at each checkpoint
def test_short_dqn_training_saves_files_and_updates_weights(tmp_path):
    models = tmp_path / "models"
    results_dir = tmp_path / "results"
    results = run_experiment(
        checkpoints=(20, 40), opponent_name="Random", seed=11,
        batch_size=4, target_update=2, learn_every=2,
        model_dir=models, output_dir=results_dir
    )
    assert len(results) == 2
    assert results[-1]["hands"] == 40
    assert results[-1]["learning_updates"] > 0
    assert 0 <= results[-1]["visited_info_sets"] <= 12

    policy_path = models / "dqn_random.json"
    strategy = load_strategy(policy_path)
    assert len(strategy) == 12
    assert all(sum(p.values()) == 1 for p in strategy.values())

    network = DQNNetwork()
    network.load_state_dict(torch.load(models / "dqn_random.pt", weights_only=True))
    assert network(torch.zeros(7)).shape == (2,)

    metadata = json.loads((models / "dqn_random_metadata.json").read_text())
    assert metadata["training_opponent"] == "Random"
    assert metadata["training_hands"] == 40
    assert metadata["learning_updates"] == results[-1]["learning_updates"]
    with (results_dir / "dqn_random.csv").open() as file:
        assert len(list(csv.DictReader(file))) == 2


# training against a mixture must produce distinct filenames
def test_mixed_training_uses_independent_filename(tmp_path):
    run_experiment(
        checkpoints=(20,), opponent_name="Mixed", seed=11,
        batch_size=4, learn_every=2,
        model_dir=tmp_path, output_dir=tmp_path
    )
    assert (tmp_path / "dqn_mixed.json").is_file()
    assert (tmp_path / "dqn_mixed.pt").is_file()


@pytest.mark.parametrize("kwargs", [
    {"checkpoints": ()},
    {"checkpoints": (20, 10)},
    {"checkpoints": (9,)},
    {"checkpoints": (-2, 20)},
    {"opponent_name": "Unknown"},
    {"learn_every": 0}
])
def test_invalid_training_configuration(kwargs, tmp_path):
    with pytest.raises(ValueError):
        options = {"checkpoints": (20,), "model_dir": tmp_path, "output_dir": tmp_path}
        options.update(kwargs)
        run_experiment(**options)

# mse must save distinct files and record which loss produced the policy
def test_mse_training_saves_without_overwriting_huber(tmp_path):
    models = tmp_path / "models"
    output = tmp_path / "results"
    huber = run_experiment(
        checkpoints=(20,), opponent_name="Mixed", loss="huber", seed=7,
        batch_size=4, learn_every=2, model_dir=models, output_dir=output
    )
    original = (models / "dqn_mixed.json").read_bytes()

    mse = run_experiment(
        checkpoints=(20,), opponent_name="Mixed", loss="mse", seed=7,
        batch_size=4, learn_every=2, model_dir=models, output_dir=output
    )

    assert (models / "dqn_mixed.json").read_bytes() == original
    assert (models / "dqn_mixed_mse.json").is_file()
    assert (models / "dqn_mixed_mse.pt").is_file()
    assert (output / "dqn_mixed_mse.csv").is_file()
    metadata = json.loads((models / "dqn_mixed_mse_metadata.json").read_text())
    assert metadata["loss"] == "mse"
    assert mse[0]["ev_vs_mixed"] == pytest.approx(
        (mse[0]["ev_vs_passive"] + mse[0]["ev_vs_random"] + mse[0]["ev_vs_nash"]) / 3
    )
    assert mse[0]["q_b_action"] in ("call", "fold")
    assert isinstance(mse[0]["q_b_gap"], float)
    assert "q_b_gap" in next(csv.DictReader((output / "dqn_mixed_mse.csv").open()))
    assert "ev_vs_mixed" in huber[0]


# keep legacy filenames and ensure invalid loss does not produce model files
def test_train_dqn_rejects_unknown_loss(tmp_path):
    with pytest.raises(ValueError, match="loss"):
        run_experiment(
            checkpoints=(20,), opponent_name="Mixed", loss="invalid",
            model_dir=tmp_path, output_dir=tmp_path
        )
    assert not list(tmp_path.iterdir())
