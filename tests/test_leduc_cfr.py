
import pytest

from poker_ai.algorithms.leduc_cfr import (
    LeducCFRTrainer,
    LeducCFRPlusTrainer
)
from poker_ai.evaluation.leduc_evaluate import information_sets


# both algorithms should discover every leduc information set
@pytest.mark.parametrize(
    "trainer_class",
    [LeducCFRTrainer, LeducCFRPlusTrainer]
)
def test_leduc_training(trainer_class):
    trainer = trainer_class()
    trainer.train(1)

    strategy = trainer.average_strategy()

    assert trainer.iteration == 1
    assert len(strategy) == len(information_sets())

    # check that each information set has a valid probability distribution
    for actions in strategy.values():
        assert sum(actions.values()) == pytest.approx(1)
        assert all(probability >= 0 for probability in actions.values())


# training should continue rather than restart when called again
@pytest.mark.parametrize(
    "trainer_class",
    [LeducCFRTrainer, LeducCFRPlusTrainer]
)
def test_training_continues(trainer_class):
    trainer = trainer_class()

    trainer.train(1)
    trainer.train(1)

    assert trainer.iteration == 2


# cfr+ must never store negative cumulative regrets
def test_leduc_cfr_plus_regret_clipping():
    trainer = LeducCFRPlusTrainer()
    trainer.train(1)

    for regrets in trainer.regret_sum.values():
        assert all(regret >= 0 for regret in regrets.values())


# invalid training budgets must be rejected
@pytest.mark.parametrize(
    "trainer_class",
    [LeducCFRTrainer, LeducCFRPlusTrainer]
)
def test_invalid_iterations(trainer_class):
    trainer = trainer_class()

    with pytest.raises(ValueError):
        trainer.train(0)
