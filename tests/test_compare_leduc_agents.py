import csv

import pytest

from poker_ai.agents.strategy_io import save_strategy
from poker_ai.evaluation.leduc_evaluate import baseline_strategy
from poker_ai.experiments.compare_leduc_agents import run_comparison


# verify that a comparison produces a valid zero sum payoff matrix
def test_comparison_matrix_and_saved_results(tmp_path):
    model_dir = tmp_path / "models"
    output_dir = tmp_path / "results"

    # create two simple strategies for an isolated test
    # this avoids depending on the actual trained models
    save_strategy(
        baseline_strategy("Random"),
        model_dir / "cfr_plus.json"
    )
    save_strategy(
        baseline_strategy("Passive"),
        model_dir / "q_learning_mixed.json"
    )

    matrix, summaries = run_comparison(
        model_dir=model_dir,
        output_dir=output_dir,
        model_files={
            "CFR+": "cfr_plus.json",
            "Q-learning Mixed": "q_learning_mixed.json"
        }
    )

    # a strategy has zero seat balanced expected payoff against itself
    assert matrix["CFR+"]["CFR+"] == 0
    assert matrix["Q-learning Mixed"]["Q-learning Mixed"] == 0

    # reverse matchups must have opposite payoffs
    assert matrix["CFR+"]["Q-learning Mixed"] == pytest.approx(
        -matrix["Q-learning Mixed"]["CFR+"]
    )

    # the tournament should produce two summary rows
    assert len(summaries) == 2

    # verify that both output files have been created
    matrix_path = output_dir / "head_to_head_matrix.csv"
    summary_path = output_dir / "agent_summary.csv"

    assert matrix_path.exists()
    assert summary_path.exists()

    with matrix_path.open() as file:
        assert len(list(csv.DictReader(file))) == 2

    with summary_path.open() as file:
        assert len(list(csv.DictReader(file))) == 2


# missing model files should produce a clear error
def test_missing_model_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing saved Leduc"):
        run_comparison(
            model_dir=tmp_path,
            output_dir=tmp_path / "results"
        )
