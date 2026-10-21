import csv

import pytest
import torch

from poker_ai.agents.dqn_agent import DQNAgent
from poker_ai.experiments.dqn_qb_diagnostic import (
    inspect_queen_bet, queen_call_counterfactual, run_diagnostic, summarise_runs
)
from poker_ai.experiments.plot_dqn_qb_diagnostic import plot_diagnostic
from poker_ai.experiments.train_dqn import run_experiment as train_dqn
from poker_ai.experiments.agent_tournament import baseline_strategy
from poker_ai.agents.nash_agent import NashAgent
from poker_ai.evaluation.matches import balanced_exact_ev


# verify the first network output means call and second means fold at q|b
def test_q_values_and_visitation_counts():
    learner = DQNAgent(seed=7)
    with torch.no_grad():
        for parameter in learner.policy_net.parameters():
            parameter.zero_()
        learner.policy_net.network[-1].bias[:] = torch.tensor([0.3, 0.7])

    info = inspect_queen_bet(learner, {"Q|b": {"call": 3, "fold": 7}})
    assert info["q_call"] == pytest.approx(0.3)
    assert info["q_fold"] == pytest.approx(0.7)
    assert info["q_call_minus_fold"] == pytest.approx(-0.4)
    assert info["q_greedy_action"] == "fold"
    assert info["q_call_rate"] == pytest.approx(0.3)
    assert info["q_visits"] == 10


# never report a visit rate if the learner has not encountered the state
def test_unvisited_queen_bet():
    info = inspect_queen_bet(DQNAgent(seed=3), {})
    assert info["q_visits"] == 0
    assert info["q_call_rate"] is None


# counterfactual must leave the original strategy unchanged
def test_counterfactual_call_is_exact_and_not_mutating():
    opponents = {
        "Passive": baseline_strategy(),
        "Random": baseline_strategy(random_play=True),
        "Nash": NashAgent().strategy
    }
    policy = baseline_strategy()
    original = {k: v.copy() for k, v in policy.items()}
    gain = queen_call_counterfactual(policy, opponents)
    modified = {k: v.copy() for k, v in policy.items()}
    modified["Q|b"] = {"call": 1.0, "fold": 0.0}
    expected = sum(balanced_exact_ev(modified, p) - balanced_exact_ev(policy, p)
                   for p in opponents.values()) / 3
    assert gain == pytest.approx(expected)
    assert policy == original


# fast smoke test for all outputs, separate model paths and plots
def test_diagnostic_saves_all_runs_without_overwriting(tmp_path):
    models = tmp_path / "models"
    output = tmp_path / "results"
    models.mkdir()
    prior = models / "dqn_mixed.json"
    prior.write_text("previous agent", encoding="utf-8")
    runs, summary = run_diagnostic(
        checkpoints=(20, 40), seeds=(2, 3),
        configs={
            "Baseline": {"learning_rate": 0.001, "epsilon": 0.1},
            "More exploration": {"learning_rate": 0.001, "epsilon": 0.3}
        },
        learn_every=4,
        model_dir=models, output_dir=output, verbose=False
    )
    assert len(runs) == 8
    assert len(summary) == 4
    assert prior.read_text() == "previous agent"
    for row in runs:
        assert row["q_visits"] == row["q_call_chosen"] + row["q_fold_chosen"]
        assert row["q_greedy_action"] in ("call", "fold")
        assert row["q_call_minus_fold"] == pytest.approx(row["q_call"] - row["q_fold"])
        assert row["q_call_gain"] >= -1e-10 or row["q_greedy_action"] == "call"
    for config in ("baseline", "more_exploration"):
        for seed in (2, 3):
            folder = models / "dqn_qb_diagnostic" / f"{config}_seed_{seed}"
            assert (folder / "policy.json").is_file()
            assert (folder / "network.pt").is_file()
            assert (folder / "metadata.json").is_file()
    with (output / "dqn_qb_diagnostic_runs.csv").open() as file:
        assert len(list(csv.DictReader(file))) == 8
    plot_diagnostic(output)
    for name in ("dqn_qb_q_gap.png", "dqn_qb_ev_mixed.png", "dqn_qb_call_rate.png"):
        assert (output / name).is_file()


# the original baseline configuration should reproduce train_dqn under the same seed
def test_baseline_matches_existing_trainer(tmp_path):
    checkpoints = (20, 40)
    rows, _ = run_diagnostic(
        checkpoints=checkpoints, seeds=(4,),
        configs={"Baseline": {"learning_rate": 0.001, "epsilon": 0.1}},
        learn_every=4,
        model_dir=tmp_path / "diagnostic_models",
        output_dir=tmp_path / "diagnostic_results", verbose=False
    )
    old = train_dqn(
        checkpoints=checkpoints, opponent_name="Mixed", seed=4,
        learn_every=4,
        model_dir=tmp_path / "original_models",
        output_dir=tmp_path / "original_results"
    )
    for new, previous in zip(rows, old):
        for metric in ("ev_vs_passive", "ev_vs_random", "ev_vs_nash", "exploitability"):
            assert new[metric] == pytest.approx(previous[metric])


# means and sample sds must be computed across seeds at the same checkpoint
def test_summary_aggregates_matched_seeds():
    rows = []
    for seed, value in ((1, 0.2), (2, 0.4)):
        rows.append({
            "configuration": "Baseline", "hands": 100, "seed": seed,
            "ev_vs_mixed": value, "ev_vs_random": value,
            "exploitability": value, "q_call_minus_fold": -0.1,
            "q_call_rate": 0.5, "q_call_chosen": 5,
            "q_fold_chosen": 5, "q_visits": 10,
            "q_call_gain": 0.0278, "q_call_locally_optimal": 0,
            "q_greedy_action": "fold"
        })
    result = summarise_runs(rows, (100,), {"Baseline": {}})[0]
    assert result["ev_vs_mixed_mean"] == pytest.approx(0.3)
    assert result["ev_vs_mixed_sd"] == pytest.approx(0.141421356)
    assert result["q_call_seeds"] == 0


# validate experiment arguments before starting expensive training
@pytest.mark.parametrize("override", [
    {"checkpoints": ()}, {"checkpoints": (40, 20)},
    {"checkpoints": (11,)}, {"seeds": ()}, {"seeds": (1, 1)},
    {"learn_every": 0}, {"configs": {}},
    {"configs": {"bad": {"learning_rate": 0, "epsilon": 0.1}}},
    {"configs": {"bad": {"learning_rate": 0.001, "epsilon": 1.1}}},
    {"configs": {"bad": {"learning_rate": 0.001}}},
    {"configs": {"bad": {"learning_rate": 0.001, "epsilon": 0.1, "loss": "invalid"}}}
])
def test_bad_arguments(override, tmp_path):
    kwargs = dict(
        checkpoints=(20,), seeds=(2,),
        configs={"Baseline": {"learning_rate": 0.001, "epsilon": 0.1}},
        output_dir=tmp_path, model_dir=tmp_path, verbose=False
    )
    kwargs.update(override)
    with pytest.raises(ValueError):
        run_diagnostic(**kwargs)


# mse configuration should be exposed without changing the standard dqn agent
def test_squared_error_loss_configuration(tmp_path):
    runs, _ = run_diagnostic(
        checkpoints=(20, 40), seeds=(3,),
        configs={
            "Squared error loss": {"learning_rate": 0.001, "epsilon": 0.1, "loss": "mse"}
        },
        model_dir=tmp_path / "models", output_dir=tmp_path / "results",
        learn_every=4, verbose=False
    )
    assert len(runs) == 2
    assert all(row["loss"] == "mse" for row in runs)
    assert (tmp_path / "models" / "dqn_qb_diagnostic" / "squared_error_loss_seed_3" / "network.pt").is_file()
