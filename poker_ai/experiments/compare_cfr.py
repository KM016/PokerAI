import csv
import time
from pathlib import Path

from poker_ai.algorithms.cfr import CFRTrainer
from poker_ai.algorithms.cfr_plus import CFRPlusTrainer
from poker_ai.evaluation.evaluate import exploitability


# compares the convergence and training time of vanilla CFR and CFR+
def run_benchmark(
    checkpoints=(
        100, 200, 500,
        1_000, 2_000, 5_000,
        10_000, 20_000, 50_000,
        100_000
    ),
    output_path="results/kuhn/cfr_comparison.csv"
):
    # store the results of both algorithms
    results = []

    # create separate trainers so each algorithm starts with no previous training
    algorithms = {
        "CFR": CFRTrainer(),
        "CFR+": CFRPlusTrainer(averaging_delay=0)
    }

    # train each algorithm independently
    for name, trainer in algorithms.items():
        print(f"\n--- {name} ---")

        previous = 0
        total_time = 0.0

        for target in checkpoints:
            # measure the time taken to reach the next checkpoint
            start = time.perf_counter()

            # continue training from the previous checkpoint rather than starting again
            trainer.train(target - previous)

            # add the additional training time to the total
            total_time += time.perf_counter() - start

            # get the average strategy learned so far
            strategy = trainer.average_strategy()

            # calculate how close the strategy is to Nash equilibrium
            # lower exploitability means the strategy is harder to exploit
            score = exploitability(strategy)

            # record exploitability and cumulative training time for each checkpoint
            results.append({
                "algorithm": name,
                "iterations": target,
                "exploitability": score,
                "training_seconds": total_time
            })

            print(
                f"{target:,} iterations | "
                f"Exploitability: {score:.8f} | "
                f"Time: {total_time:.3f}s"
            )

            # update the previous checkpoint so we only train the additional iterations
            previous = target

    # create the results folder if it does not already exist
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # save the results as a CSV for plotting and comparing both algorithms
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "algorithm",
                "iterations",
                "exploitability",
                "training_seconds"
            ]
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    run_benchmark()
