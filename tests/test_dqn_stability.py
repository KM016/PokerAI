import csv

import pytest

from poker_ai.experiments.dqn_stability import run_stability, summarise_runs
from poker_ai.experiments.plot_dqn_stability import plot_stability


# verify mixed ev is calculated from the three equally likely opponents
def test_stability_runs_and_saves_independent_models(tmp_path):
    model_dir = tmp_path / "models"
    output_dir = tmp_path / "results"
    model_dir.mkdir()
    output_dir.mkdir()

    # check that stability training cannot overwrite the original dqn model
    old_model = model_dir / "dqn_mixed.json"
    old_model.write_text("original model")

    runs, summary = run_stability(
        checkpoints=(20, 40), seeds=(2, 3),
        opponents=("Random", "Mixed"), learn_every=4,
        model_dir=model_dir, output_dir=output_dir, verbose=False
    )

    assert len(runs) == 8
    assert len(summary) == 4
    assert all(row["n_seeds"] == 2 for row in summary)
    assert old_model.read_text() == "original model"
    assert all(0 <= row["visited_info_sets"] <= 12 for row in runs)

    for row in runs:
        assert row["ev_vs_mixed"] == pytest.approx(
            (row["ev_vs_passive"] + row["ev_vs_random"] + row["ev_vs_nash"]) / 3
        )

    for opponent in ("Random", "Mixed"):
        for seed in (2, 3):
            dirname = f"{opponent.lower()}_seed_{seed}_every_4"
            saved = model_dir / "dqn_stability" / dirname
            assert (saved / f"dqn_{opponent.lower()}.json").is_file()
            assert (saved / f"dqn_{opponent.lower()}.pt").is_file()
            assert (saved / f"dqn_{opponent.lower()}_metadata.json").is_file()

    with (output_dir / "dqn_stability_runs.csv").open() as file:
        assert len(list(csv.DictReader(file))) == 8
    assert (output_dir / "dqn_stability_summary.csv").is_file()

    plot_stability(output_dir)
    assert (output_dir / "dqn_stability_ev_mixed.png").is_file()
    assert (output_dir / "dqn_stability_ev_random.png").is_file()
    assert (output_dir / "dqn_stability_exploitability.png").is_file()


# mean and sample standard deviation must be computed over seeds, not checkpoints
def test_summary_aggregates_matching_seeds():
    rows = []
    for seed, ev in ((1, 0.2), (2, 0.4)):
        rows.append({
            "training_opponent": "Mixed", "seed": seed, "hands": 20,
            "ev_vs_random": ev, "ev_vs_mixed": ev / 2,
            "ev_vs_nash": 0.0, "exploitability": 0.25,
            "visited_info_sets": 12, "learning_updates": 5,
            "training_seconds": 1.0
        })

    summary = summarise_runs(rows, (20,), ("Mixed",))
    assert len(summary) == 1
    assert summary[0]["ev_vs_random_mean"] == pytest.approx(0.3)
    assert summary[0]["ev_vs_random_sd"] == pytest.approx(0.1414213562)
    assert summary[0]["ev_vs_mixed_mean"] == pytest.approx(0.15)
    assert summary[0]["exploitability_sd"] == 0.0


# reject invalid experiment settings before any training begins
@pytest.mark.parametrize("options", [
    {"checkpoints": ()},
    {"checkpoints": (40, 20)},
    {"checkpoints": (21,)},
    {"seeds": ()},
    {"seeds": (2, 2)},
    {"opponents": ()},
    {"opponents": ("Passive",)},
    {"opponents": ("Random", "Random")},
    {"learn_every": 0}
])
def test_invalid_stability_options(options, tmp_path):
    kwargs = {
        "checkpoints": (20,), "seeds": (2,), "opponents": ("Random",),
        "model_dir": tmp_path, "output_dir": tmp_path, "verbose": False
    }
    kwargs.update(options)
    with pytest.raises(ValueError):
        run_stability(**kwargs)


# exact evaluations should repeat for the same seed when starting from scratch
def test_stability_is_reproducible(tmp_path):
    kwargs = {
        "checkpoints": (20, 40), "seeds": (6,),
        "opponents": ("Mixed",), "learn_every": 4, "verbose": False
    }
    a, _ = run_stability(
        model_dir=tmp_path / "models_a", output_dir=tmp_path / "results_a", **kwargs
    )
    b, _ = run_stability(
        model_dir=tmp_path / "models_b", output_dir=tmp_path / "results_b", **kwargs
    )
    for left, right in zip(a, b):
        for metric in ("ev_vs_random", "ev_vs_mixed", "ev_vs_nash", "exploitability"):
            assert left[metric] == pytest.approx(right[metric])
