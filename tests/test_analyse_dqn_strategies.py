import csv
from copy import deepcopy

import pytest

from poker_ai.agents.nash_agent import NashAgent
from poker_ai.agents.strategy_io import save_strategy
from poker_ai.experiments.agent_tournament import baseline_strategy
from poker_ai.experiments.analyse_dqn_strategies import (
    INFO_SETS,
    alternate_action_strategy,
    changed_action_strategy,
    chosen_action,
    measure_strategy,
    optimal_mixed_strategy,
    run_analysis,
    saved_policy_path,
    validate_strategy
)
from poker_ai.experiments.plot_dqn_strategy_analysis import plot_strategies


def opponents():
    return {
        "Passive": baseline_strategy(),
        "Random": baseline_strategy(random_play=True),
        "Nash": NashAgent().strategy
    }


def test_validates_complete_kuhn_strategy():
    valid = baseline_strategy()
    validate_strategy(valid)
    incomplete = deepcopy(valid)
    del incomplete["J|"]
    with pytest.raises(ValueError, match="12 Kuhn"):
        validate_strategy(incomplete)


@pytest.mark.parametrize("bad", [
    {"check": 0.7, "bet": 0.7},
    {"check": -0.1, "bet": 1.1},
    {"check": float("nan"), "bet": 1.0},
    {"check": 1.0, "fold": 0.0}
])
def test_rejects_invalid_action_distributions(bad):
    strategy = baseline_strategy()
    strategy["J|"] = bad
    with pytest.raises(ValueError):
        validate_strategy(strategy)


def test_action_selection_requires_a_greedy_policy():
    strategy = baseline_strategy()
    assert chosen_action(strategy, "J|") == "check"
    assert chosen_action(strategy, "K|b") == "fold"
    strategy["J|"] = {"check": 0.5, "bet": 0.5}
    with pytest.raises(ValueError, match="deterministic"):
        chosen_action(strategy, "J|")


def test_saved_policy_paths_match_stability_filenames(tmp_path):
    path = saved_policy_path(tmp_path, "Mixed", 44, 16)
    assert path == (
        tmp_path / "dqn_stability" / "mixed_seed_44_every_16" / "dqn_mixed.json"
    )


def test_optimal_mixed_response_is_computed_exactly():
    optimal_policy, value = optimal_mixed_strategy(opponents())
    validate_strategy(optimal_policy)
    assert value == pytest.approx(103 / 216)
    assert measure_strategy(optimal_policy, opponents())["Mixed"] == pytest.approx(value)


def test_one_action_swaps_do_not_change_original_strategy():
    source = baseline_strategy()
    alternative = deepcopy(source)
    alternative["J|"] = {"check": 0.0, "bet": 1.0}
    swapped = changed_action_strategy(source, "J|", alternative)
    assert chosen_action(swapped, "J|") == "bet"
    assert chosen_action(source, "J|") == "check"
    assert all(swapped[info] == source[info] for info in INFO_SETS if info != "J|")
    flipped = alternate_action_strategy(source, "K|cb")
    assert chosen_action(flipped, "K|cb") == "call"
    assert chosen_action(source, "K|cb") == "fold"


def test_exact_mixed_ev_is_average_of_component_evs():
    scores = measure_strategy(baseline_strategy(), opponents())
    assert scores["Mixed"] == pytest.approx(
        (scores["Passive"] + scores["Random"] + scores["Nash"]) / 3
    )
    assert scores["exploitability"] >= 0
    assert "exploitability" not in measure_strategy(
        baseline_strategy(), opponents(), with_exploitability=False
    )


def save_pair(model_dir, seed, random_policy, mixed_policy):
    for name, policy in (("Random", random_policy), ("Mixed", mixed_policy)):
        save_strategy(policy, saved_policy_path(model_dir, name, seed, 16))


def test_analysis_is_read_only_and_explains_action_costs(tmp_path):
    model_dir = tmp_path / "models"
    out = tmp_path / "results"
    random_policy = baseline_strategy()
    mixed_policy = baseline_strategy()
    random_policy["J|"] = {"check": 0.0, "bet": 1.0}
    random_policy["Q|b"] = {"call": 1.0, "fold": 0.0}
    save_pair(model_dir, 42, random_policy, mixed_policy)
    original_bytes = saved_policy_path(model_dir, "Mixed", 42, 16).read_bytes()

    metrics, actions, differences, opportunities = run_analysis(
        seeds=(42,), model_dir=model_dir, output_dir=out, verbose=False
    )
    assert len(metrics) == 2
    assert len(actions) == 12
    assert all("optimal_action" in row for row in actions)
    assert metrics[0]["gap_to_mixed_optimum"] >= -1e-9
    assert len(differences) == 2
    assert len(opportunities) == 12
    assert saved_policy_path(model_dir, "Mixed", 42, 16).read_bytes() == original_bytes
    assert (out / "dqn_strategy_report.md").is_file()
    assert "### Seed 42" in (out / "dqn_strategy_report.md").read_text()
    assert all((out / name).is_file() for name in (
        "dqn_strategy_actions.csv", "dqn_strategy_metrics.csv",
        "dqn_strategy_differences.csv", "dqn_strategy_opportunities.csv"
    ))
    changes = {r["info_set"]: r for r in differences}
    assert set(changes) == {"J|", "Q|b"}
    swapped = changed_action_strategy(mixed_policy, "J|", random_policy)
    before = measure_strategy(mixed_policy, opponents(), with_exploitability=False)
    after = measure_strategy(swapped, opponents(), with_exploitability=False)
    assert changes["J|"]["delta_ev_vs_mixed_if_switched"] == pytest.approx(
        after["Mixed"] - before["Mixed"]
    )
    with (out / "dqn_strategy_actions.csv").open() as file:
        assert len(list(csv.DictReader(file))) == 12


def test_report_covers_every_seed_and_handles_no_disagreements(tmp_path):
    model_dir = tmp_path / "models"
    for seed in (42, 43):
        save_pair(model_dir, seed, baseline_strategy(), baseline_strategy())
    metrics, actions, differences, opportunities = run_analysis(
        seeds=(42, 43), model_dir=model_dir, output_dir=tmp_path / "results",
        verbose=False
    )
    assert len(metrics) == 4
    assert len(actions) == 24
    assert len(opportunities) == 24
    assert differences == []
    with (tmp_path / "results" / "dqn_strategy_differences.csv").open() as file:
        assert len(list(csv.DictReader(file))) == 0
    path = plot_strategies(tmp_path / "results")
    assert path.is_file()
    assert path.stat().st_size > 5000


@pytest.mark.parametrize("params", [
    {"seeds": ()},
    {"seeds": (42, 42)},
    {"learn_every": 0}
])
def test_rejects_invalid_analysis_configuration(tmp_path, params):
    with pytest.raises(ValueError):
        run_analysis(model_dir=tmp_path, output_dir=tmp_path, verbose=False, **params)


def test_missing_saved_model_has_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="dqn_stability first"):
        run_analysis(seeds=(42,), model_dir=tmp_path / "models",
                     output_dir=tmp_path / "results", verbose=False)
