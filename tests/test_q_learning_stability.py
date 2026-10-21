import csv

import pytest

from poker_ai.experiments.q_learning_stability import (
    learning_rates, run_experiment
)
from poker_ai.experiments.plot_q_learning_stability import plot_stability


def test_constant_rates_do_not_change():
    assert learning_rates("Constant", 99, 100) == (0.1, 0.1)


def test_decay_changes_only_selected_parameters():
    alpha, epsilon = learning_rates("Decay alpha", 99, 100)
    assert alpha == pytest.approx(0.01)
    assert epsilon == 0.1

    alpha, epsilon = learning_rates("Decay epsilon", 99, 100)
    assert alpha == 0.1
    assert epsilon == pytest.approx(0.01)

    assert learning_rates("Decay both", 0, 100) == (0.1, 0.1)
    assert learning_rates("Decay both", 99, 100) == pytest.approx((0.01, 0.01))


def test_stability_results_have_correct_seed_counts(tmp_path):
    runs, summary = run_experiment(
        checkpoints=(10, 20),
        seeds=(5, 6),
        configurations=("Constant", "Decay both"),
        output_dir=tmp_path,
        verbose=False
    )

    assert len(runs) == 8
    assert len(summary) == 4
    assert all(row["seeds"] == 2 for row in summary)
    assert all(0 <= row["visited_info_sets"] <= 12 for row in runs)
    assert (tmp_path / "q_learning_stability_runs.csv").exists()
    assert (tmp_path / "q_learning_stability_summary.csv").exists()

    with (tmp_path / "q_learning_stability_runs.csv").open() as file:
        assert len(list(csv.DictReader(file))) == 8

    plot_stability(output_dir=tmp_path)
    assert (tmp_path / "q_learning_stability_ev.png").exists()
    assert (tmp_path / "q_learning_stability_exploitability.png").exists()


def test_stability_reproducible_with_same_seeds(tmp_path):
    kwargs = dict(checkpoints=(10, 20), seeds=(7, 8),
                  configurations=("Decay alpha",), verbose=False)
    first, _ = run_experiment(output_dir=tmp_path / "a", **kwargs)
    second, _ = run_experiment(output_dir=tmp_path / "b", **kwargs)
    assert first == second
