import csv

from poker_ai.experiments.compare_q_learning_opponents import run_comparison


def test_comparison_trains_all_opponents(tmp_path, monkeypatch):
    # use a temporary directory to avoid overwriting real experiments
    monkeypatch.chdir(tmp_path)
    results = run_comparison(checkpoints=(2, 4), seed=7)

    # four training modes each measured at two checkpoints
    assert len(results) == 8
    assert {row["training_opponent"] for row in results} == {
        "Passive", "Random", "Nash", "Mixed"
    }
    assert all(0 <= row["visited_info_sets"] <= 12 for row in results)

    # check that the saved models and result data exist
    for mode in ("passive", "random", "nash", "mixed"):
        assert (tmp_path / f"models/kuhn/q_learning_{mode}.json").exists()

    output = tmp_path / "results/kuhn/q_learning_opponents.csv"
    with output.open(newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 8
