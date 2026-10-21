import csv

import pytest

from poker_ai.agents.strategy_io import save_strategy
from poker_ai.evaluation.leduc_evaluate import baseline_strategy
from poker_ai.experiments.leduc_monte_carlo import (
    simulate_session,
    run_experiment
)


# identical passive agents should have zero balanced profit
# each card deal is played once from each position
def test_identical_passive_agents():
    passive = baseline_strategy("Passive")

    result = simulate_session(
        passive, passive, hands=100, seed=42
    )

    assert result == pytest.approx(0.0)


# repeating the same experiment with the same seed must give the same result
def test_reproducible_session():
    random_policy = baseline_strategy("Random")
    passive = baseline_strategy("Passive")

    first = simulate_session(
        random_policy, passive, hands=100, seed=42
    )
    second = simulate_session(
        random_policy, passive, hands=100, seed=42
    )

    assert first == second


# invalid session sizes should be rejected
def test_invalid_session_size():
    policy = baseline_strategy("Random")

    with pytest.raises(ValueError):
        simulate_session(policy, policy, hands=99)


# verify that a small experiment produces both csv files and both plots
def test_monte_carlo_outputs(tmp_path):
    model_dir = tmp_path / "models"
    output_dir = tmp_path / "results"

    # create two temporary policies without needing trained model files
    save_strategy(
        baseline_strategy("Passive"),
        model_dir / "cfr.json"
    )
    save_strategy(
        baseline_strategy("Random"),
        model_dir / "cfr_plus.json"
    )

    sessions, summaries = run_experiment(
        model_dir=model_dir,
        output_dir=output_dir,
        model_files={
            "CFR": "cfr.json",
            "CFR+": "cfr_plus.json"
        },
        hands_per_session=20
    )

    # one matchup with 20 independent sessions
    assert len(sessions) == 20
    assert len(summaries) == 1

    # the confidence interval must contain its estimated mean
    result = summaries[0]

    assert result["ci_lower_95"] <= result["simulated_ev"]
    assert result["simulated_ev"] <= result["ci_upper_95"]

    # both statistical data files should contain the expected number of rows
    with (output_dir / "monte_carlo_sessions.csv").open() as file:
        assert len(list(csv.DictReader(file))) == 20

    with (output_dir / "monte_carlo_summary.csv").open() as file:
        assert len(list(csv.DictReader(file))) == 1

    # both plots must be successfully generated
    assert (output_dir / "monte_carlo_comparison.png").exists()
    assert (output_dir / "monte_carlo_cfr_variance.png").exists()
