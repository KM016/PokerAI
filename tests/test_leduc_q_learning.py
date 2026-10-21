import pytest

from poker_ai.agents.strategy_io import load_strategy, save_strategy
from poker_ai.algorithms.q_learning import QLearningAgent
from poker_ai.evaluation.leduc_evaluate import (
    baseline_strategy,
    information_sets
)
from poker_ai.experiments.train_q_learning import (
    greedy_strategy,
    run_experiment
)


# a leduc q learning policy must include every possible information set
def test_leduc_greedy_strategy():
    learner = QLearningAgent()

    # no training has occurred so the q table should remain empty
    strategy = greedy_strategy(learner, game_name="leduc")

    assert len(strategy) == len(information_sets())
    assert learner.q_table == {}

    # all legal actions should have valid probabilities
    for actions in strategy.values():
        assert sum(actions.values()) == pytest.approx(1)


# check that leduc training saves the policy, q values and results
def test_leduc_training(tmp_path):
    model_dir = tmp_path / "models"
    output_dir = tmp_path / "results"

    # create a temporary reference policy for this isolated test
    # the real training experiment uses our saved cfr+ strategy
    save_strategy(
        baseline_strategy("Random"),
        model_dir / "cfr_plus.json"
    )

    # train a very small number of hands to test the full training pipeline
    results = run_experiment(
        game="leduc",
        checkpoints=(2, 4),
        opponent_name="Random",
        schedule="Decay alpha",
        seed=101,
        model_dir=model_dir,
        output_dir=output_dir
    )

    filename = "q_learning_random_decay_alpha"

    policy = load_strategy(model_dir / f"{filename}.json")
    q_values = load_strategy(model_dir / f"{filename}_q_values.json")

    assert len(results) == 2
    assert len(policy) == len(information_sets())
    assert q_values
    assert 0 < results[-1]["visited_info_sets"] <= len(information_sets())

    assert (model_dir / f"{filename}_metadata.json").exists()
    assert (output_dir / f"{filename}.csv").exists()
