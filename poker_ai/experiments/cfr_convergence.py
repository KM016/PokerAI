import csv
import time
from pathlib import Path

from poker_ai.algorithms.cfr import CFRTrainer
from poker_ai.evaluation.evaluate import exploitability


# trains vanilla CFR and measures how exploitability changes with more iterations
def run_experiment(
    checkpoints=(
        100, 200, 500,
        1_000, 2_000, 5_000,
        10_000, 20_000, 50_000,
        100_000
    ),
    output_path="results/kuhn/cfr_convergence.csv"
):
    # create a new CFR trainer with no previous training
    trainer = CFRTrainer()

    # store exploitability and training time at each checkpoint
    results = []
    previous = 0
    total_training_time = 0.0

    for target in checkpoints:
        # start measuring how long the next set of iterations takes
        start = time.perf_counter()

        # continue training from the previous checkpoint instead of starting again
        trainer.train(target - previous)

        # add the new training time to the total
        total_training_time += time.perf_counter() - start

        # get the average strategy learned so far
        strategy = trainer.average_strategy()

        # calculate how far the strategy is from Nash equilibrium
        # lower exploitability means the strategy is harder to exploit
        score = exploitability(strategy)

        # record the results so we can compare different training stages
        results.append({
            "iterations": target,
            "exploitability": score,
            "training_seconds": total_training_time
        })

        print(f"Iterations: {target:,}")
        print(f"Exploitability: {score:.8f}")
        print(f"Training time: {total_training_time:.3f}s\n")

        # remember the checkpoint so we only train the extra iterations next time
        previous = target

    # create the results folder if it does not already exist
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # save the results as a CSV so we can use them to generate graphs
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "iterations",
                "exploitability",
                "training_seconds"
            ]
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"Results saved to: {output_path}")

    return results


if __name__ == "__main__":
    run_experiment()
