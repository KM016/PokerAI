import csv

import pytest

from poker_ai.agents.nash_agent import NashAgent
from poker_ai.agents.strategy_io import save_strategy
from poker_ai.experiments.agent_tournament import baseline_strategy
from poker_ai.experiments.kuhn_final_benchmark import (
    build_metrics, head_to_head, load_models, optimal_mixed_ev,
    run_benchmark, validate_strategy
)
from poker_ai.experiments.plot_kuhn_final_benchmark import plot_benchmark


def opponents():
    return {
        "Passive": baseline_strategy(),
        "Random": baseline_strategy(random_play=True),
        "Nash": NashAgent().strategy
    }


# the reference opponent mixture has an independently checkable exact optimum
def test_optimal_mixed_ev():
    assert optimal_mixed_ev(opponents()) == pytest.approx(103 / 216)


# all twelve information sets must have valid action probability distributions
def test_invalid_strategy_is_rejected():
    strategy = baseline_strategy()
    strategy["Q|b"] = {"call": 0.7, "fold": 0.2}
    with pytest.raises(ValueError, match="sum"):
        validate_strategy(strategy)


def test_missing_information_set_is_rejected():
    strategy = baseline_strategy()
    del strategy["J|"]
    with pytest.raises(ValueError, match="missing"):
        validate_strategy(strategy)


def test_nash_is_always_available_even_without_model_files(tmp_path):
    strategies, metadata, sources, missing = load_models(tmp_path)
    assert list(strategies) == ["Nash"]
    assert len(missing) == 7


# a saved q learning model is loaded rather than retrained
def test_loads_saved_policy(tmp_path):
    save_strategy(baseline_strategy(), tmp_path / "q_learning_mixed_decay_alpha.json")
    strategies, metadata, sources, missing = load_models(tmp_path)
    assert "Q-learning Mixed" in strategies
    assert "q_learning_mixed_decay_alpha.json" not in missing


def test_nash_has_zero_exploitability_and_zero_ev_vs_nash():
    strategies = {"Nash": NashAgent().strategy}
    rows = build_metrics(strategies, {"Nash": {}}, {"Nash": "builtin"}, opponents())
    assert rows[0]["exploitability"] == pytest.approx(0, abs=1e-12)
    assert rows[0]["ev_vs_nash"] == pytest.approx(0, abs=1e-12)
    assert rows[0]["ev_vs_mixed"] == pytest.approx(
        (rows[0]["ev_vs_passive"] + rows[0]["ev_vs_random"] + rows[0]["ev_vs_nash"]) / 3
    )


def test_head_to_head_is_antisymmetric():
    strategies = {"Nash": NashAgent().strategy, "Passive": baseline_strategy()}
    rows = head_to_head(strategies)
    value = {(row["agent"], row["opponent"]): row["balanced_exact_ev"] for row in rows}
    assert value[("Nash", "Passive")] == pytest.approx(-value[("Passive", "Nash")])


def test_benchmark_writes_outputs(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    save_strategy(baseline_strategy(), model_dir / "cfr.json")
    output = tmp_path / "results"
    rows, pairings, missing = run_benchmark(model_dir, output)
    assert len(rows) == 2
    assert len(pairings) == 4
    assert len(missing) == 6
    assert (output / "kuhn_final_benchmark.md").exists()
    with (output / "kuhn_final_benchmark.csv").open(newline="") as file:
        assert len(list(csv.DictReader(file))) == 2
    plot_benchmark(output)
    assert (output / "kuhn_final_profitability.png").exists()
    assert (output / "kuhn_final_exploitability.png").exists()
