import argparse
import csv
import io
from contextlib import redirect_stdout
from pathlib import Path
from statistics import mean, stdev

from poker_ai.experiments.train_dqn import run_experiment as train_dqn
from poker_ai.experiments.train_q_learning import CHECKPOINTS


# compare how reliably dqn learns against random and mixed opponents
# use matched random seeds and the same training budget for each setting
OPPONENTS = ("Random", "Mixed")
SEEDS = (42, 43, 44, 45, 46)
METRICS = ("ev_vs_random", "ev_vs_mixed", "ev_vs_nash", "exploitability")


def save_csv(path, rows):
    # store the raw runs and summary separately so the graphs are reproducible
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarise_runs(runs, checkpoints, opponents):
    # each checkpoint is summarised using the five independent training seeds
    summary = []
    for opponent in opponents:
        for hands in checkpoints:
            sample = [
                row for row in runs
                if row["training_opponent"] == opponent and row["hands"] == hands
            ]
            row = {
                "training_opponent": opponent,
                "hands": hands,
                "n_seeds": len(sample)
            }
            for metric in METRICS:
                values = [result[metric] for result in sample]
                row[f"{metric}_mean"] = mean(values)
                row[f"{metric}_sd"] = stdev(values) if len(values) > 1 else 0.0
            row["coverage_mean"] = mean(r["visited_info_sets"] for r in sample)
            row["updates_mean"] = mean(r["learning_updates"] for r in sample)
            row["training_seconds_mean"] = mean(r["training_seconds"] for r in sample)
            summary.append(row)
    return summary


def run_stability(
    checkpoints=CHECKPOINTS,
    seeds=SEEDS,
    opponents=OPPONENTS,
    learn_every=16,
    output_dir="results/kuhn",
    model_dir="models/kuhn",
    verbose=True
):
    checkpoints = tuple(checkpoints)
    seeds = tuple(seeds)
    opponents = tuple(opponents)
    if not checkpoints or any(h <= 0 for h in checkpoints):
        raise ValueError("Checkpoints must be positive and nonempty")
    if tuple(sorted(set(checkpoints))) != checkpoints or checkpoints[-1] % 2:
        raise ValueError("Checkpoints must increase and end on an even hand count")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be nonempty and unique")
    if not opponents or len(set(opponents)) != len(opponents):
        raise ValueError("Opponents must be nonempty and unique")
    if any(name not in OPPONENTS for name in opponents):
        raise ValueError("Only Random and Mixed training are compared")
    if learn_every <= 0:
        raise ValueError("learn_every must be positive")

    output_dir = Path(output_dir)
    model_dir = Path(model_dir)
    runs = []

    for opponent in opponents:
        if verbose:
            print(f"\n--- DQN trained against {opponent} ---")

        for seed in seeds:
            # every seed and opponent has its own folder to avoid overwriting models
            run_name = f"{opponent.lower()}_seed_{seed}_every_{learn_every}"
            run_models = model_dir / "dqn_stability" / run_name
            run_results = output_dir / "dqn_stability_runs" / run_name

            # reuse the existing poker training logic instead of creating another trainer
            # suppress the checkpoint logs and show one concise result for each seed
            with redirect_stdout(io.StringIO()):
                checkpoints_result = train_dqn(
                    checkpoints=checkpoints,
                    opponent_name=opponent,
                    seed=seed,
                    learn_every=learn_every,
                    model_dir=run_models,
                    output_dir=run_results
                )

            for result in checkpoints_result:
                row = {
                    "training_opponent": opponent,
                    "seed": seed,
                    "learn_every": learn_every,
                    **result
                }
                # mixed randomly picks passive, random or nash with equal probability
                # linearity of expectation gives the exact ev versus that mixture
                row["ev_vs_mixed"] = (
                    row["ev_vs_passive"] + row["ev_vs_random"] + row["ev_vs_nash"]
                ) / 3
                runs.append(row)

            if verbose:
                last = runs[-1]
                print(
                    f"Seed {seed} | "
                    f"EV vs Mixed: {last['ev_vs_mixed']:+.4f} | "
                    f"EV vs Random: {last['ev_vs_random']:+.4f} | "
                    f"Exploitability: {last['exploitability']:.4f} | "
                    f"Time: {last['training_seconds']:.1f}s"
                )

    summary = summarise_runs(runs, checkpoints, opponents)
    save_csv(output_dir / "dqn_stability_runs.csv", runs)
    save_csv(output_dir / "dqn_stability_summary.csv", summary)

    if verbose:
        print("\n--- Mean results at the final checkpoint ---")
        for row in summary:
            if row["hands"] == checkpoints[-1]:
                print(
                    f"{row['training_opponent']}: "
                    f"Mixed EV {row['ev_vs_mixed_mean']:+.4f} "
                    f"± {row['ev_vs_mixed_sd']:.4f} | "
                    f"Random EV {row['ev_vs_random_mean']:+.4f} "
                    f"± {row['ev_vs_random_sd']:.4f} | "
                    f"Exploitability {row['exploitability_mean']:.4f} "
                    f"± {row['exploitability_sd']:.4f}"
                )
        print(f"\nSummary and per-seed CSV files saved to: {output_dir}")
        print(f"Independent models saved under: {model_dir / 'dqn_stability'}")

    return runs, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DQN training stability across five seeds")
    parser.add_argument("--hands", type=int, default=100_000)
    parser.add_argument("--learn-every", type=int, default=16)
    args = parser.parse_args()
    if args.hands <= 0 or args.hands % 2:
        parser.error("--hands must be a positive even number")
    checkpoints = tuple(h for h in CHECKPOINTS if h < args.hands) + (args.hands,)
    run_stability(checkpoints=checkpoints, learn_every=args.learn_every)
