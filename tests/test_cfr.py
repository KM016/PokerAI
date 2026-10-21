
import pytest
from poker_ai.algorithms.cfr import regret_matching
from poker_ai.evaluation.evaluate import exploitability


# TEST 1: Positive regrets should become proportional probabilities
def test_positive_regrets():
    regrets = {"check": 4, "bet": 6}

    strategy = regret_matching(regrets)

    assert strategy["check"] == pytest.approx(0.4)
    assert strategy["bet"] == pytest.approx(0.6)


# TEST 2: Negative regrets should be ignored
def test_negative_regrets():
    regrets = {"check": -3, "bet": 7}

    strategy = regret_matching(regrets)

    assert strategy["check"] == 0
    assert strategy["bet"] == 1


# TEST 3: All negative regrets should produce a uniform strategy
def test_all_negative_regrets():
    regrets = {"check": -3, "bet": -7}

    strategy = regret_matching(regrets)

    assert strategy["check"] == 0.5
    assert strategy["bet"] == 0.5


# TEST 4: All zero regrets should also produce a uniform strategy
def test_zero_regrets():
    regrets = {"check": 0, "bet": 0}

    strategy = regret_matching(regrets)

    assert strategy["check"] == 0.5
    assert strategy["bet"] == 0.5



from poker_ai.algorithms.cfr import CFRTrainer


# TEST 5: A new information set starts with zero regrets
def test_cfr_initialisation():
    trainer = CFRTrainer()

    strategy = trainer.get_strategy("J|", ["check", "bet"])

    # No regrets yet, so both actions have equal probability
    assert strategy == {"check": 0.5, "bet": 0.5}

    # Both tracking dictionaries should start at zero
    assert trainer.regret_sum["J|"] == {"check": 0.0, "bet": 0.0}
    assert trainer.strategy_sum["J|"] == {"check": 0.0, "bet": 0.0}


# TEST 6: The strategy should change when regrets change
def test_cfr_strategy_update():
    trainer = CFRTrainer()

    trainer.get_strategy("J|", ["check", "bet"])

    # Manually assign regrets to test regret matching
    trainer.regret_sum["J|"] = {"check": 3, "bet": 7}

    strategy = trainer.get_strategy("J|", ["check", "bet"])

    assert strategy["check"] == pytest.approx(0.3)
    assert strategy["bet"] == pytest.approx(0.7)


# TEST 7: Different information sets have separate regrets
def test_cfr_independent_information_sets():
    trainer = CFRTrainer()

    trainer.get_strategy("J|", ["check", "bet"])
    trainer.get_strategy("K|", ["check", "bet"])

    # Change the regrets for Jack only
    trainer.regret_sum["J|"]["bet"] = 10

    # King's regrets should be unaffected
    assert trainer.regret_sum["K|"] == {"check": 0.0, "bet": 0.0}



# TEST 8: Training should discover all 12 information sets
def test_training_covers_all_states():
    trainer = CFRTrainer()

    trainer.train(1)

    assert len(trainer.regret_sum) == 12
    assert len(trainer.strategy_sum) == 12


# TEST 9: Average strategy probabilities must be valid
def test_average_strategy_probabilities():
    trainer = CFRTrainer()

    trainer.train(20)

    for actions in trainer.average_strategy().values():
        # Probabilities should sum to 1
        assert sum(actions.values()) == pytest.approx(1)

        # Each probability must be between 0 and 1
        assert all(0 <= p <= 1 for p in actions.values())


# TEST 10: CFR should learn a low-exploitability strategy
def test_cfr_convergence():
    trainer = CFRTrainer()

    # Train for 1000 full game-tree iterations
    trainer.train(1000)

    # Extract the learned average strategy
    strategy = trainer.average_strategy()

    # Exploitability should be close to zero
    assert exploitability(strategy) < 0.03


# TEST 11: Reject invalid training iteration counts
@pytest.mark.parametrize("iterations", [0, -1])
def test_invalid_iterations(iterations):
    trainer = CFRTrainer()

    with pytest.raises(ValueError):
        trainer.train(iterations)


from poker_ai.experiments.cfr_convergence import run_experiment


# TEST 12: Check that the experiment records and saves results
def test_convergence_experiment(tmp_path):
    # Use a temporary output file so tests don't alter real results
    output = tmp_path / "convergence.csv"

    # Run a small experiment
    results = run_experiment(
        checkpoints=(10, 20),
        output_path=output
    )

    # Check that both checkpoints were recorded
    assert len(results) == 2
    assert [r["iterations"] for r in results] == [10, 20]

    # Exploitability should be non-negative
    assert all(r["exploitability"] >= 0 for r in results)

    # Check that the CSV file was created
    assert output.exists()

