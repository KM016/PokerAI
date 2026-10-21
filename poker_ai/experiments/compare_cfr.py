import argparse
import csv
import time
from pathlib import Path

from poker_ai.algorithms.cfr import CFRTrainer
from poker_ai.algorithms.cfr_plus import CFRPlusTrainer
from poker_ai.algorithms.leduc_cfr import LeducCFRTrainer, LeducCFRPlusTrainer
from poker_ai.evaluation.evaluate import exploitability as kuhn_exploitability
from poker_ai.evaluation.leduc_evaluate import exploitability as leduc_exploitability
from poker_ai.agents.strategy_io import save_strategy


# compare cfr and cfr+ convergence, exploitability and training time
# supports both kuhn and leduc using the same experiment
def run_benchmark(game="kuhn", checkpoints=None):

    # select the trainers and evaluator for the chosen poker game
    if game == "kuhn":
        trainers = {
            "CFR": CFRTrainer(),
            "CFR+": CFRPlusTrainer(averaging_delay=0)
        }
        evaluate = kuhn_exploitability

        # kuhn is small enough to train for many full tree iterations
        if checkpoints is None:
            checkpoints = (
                100, 200, 500, 1_000, 2_000,
                5_000, 10_000, 20_000, 50_000, 100_000
            )

    elif game == "leduc":
        trainers = {
            "CFR": LeducCFRTrainer(),
            "CFR+": LeducCFRPlusTrainer(averaging_delay=0)
        }
        evaluate = leduc_exploitability

        # start with a small budget because the leduc tree is much larger
        # we will increase this once we know the cost of each iteration
        if checkpoints is None:
            checkpoints = (
                1, 2, 5, 10, 20,
                50, 100, 200
            )

    else:
        raise ValueError("Choose either kuhn or leduc")

    # store the results for both algorithms
    results = []

    for name, trainer in trainers.items():
        print(f"\n--- {game.upper()} {name} ---")

        previous = 0
        training_time = 0.0

        for target in checkpoints:
            # continue training rather than restarting at each checkpoint
            # example: moving from 5 to 10 requires 5 extra iterations
            additional = target - previous

            # measure only the time spent training the algorithm
            start = time.perf_counter()
            trainer.train(additional)
            training_time += time.perf_counter() - start

            # retrieve the average strategy accumulated during training
            strategy = trainer.average_strategy()

            # find how vulnerable this strategy is to an optimal response
            # smaller values indicate a strategy closer to equilibrium
            score = evaluate(strategy)

            # save the measurements for later plotting and comparison
            results.append({
                "algorithm": name,
                "iterations": target,
                "exploitability": score,
                "training_seconds": training_time
            })

            print(
                f"{target:,} iterations | "
                f"Exploitability: {score:.8f} | "
                f"Training time: {training_time:.3f}s"
            )

            # remember the current checkpoint for the next training step
            previous = target

        # save the final average strategy after all training checkpoints
        # this lets us play against the agent without retraining it
        model_name = name.lower().replace("+", "_plus")
        model_path = Path(f"models/{game}/{model_name}.json")

        save_strategy(trainer.average_strategy(), model_path)
        print(f"Saved strategy to: {model_path}")

    # save results in a separate directory for the selected poker game
    output_path = Path(f"results/{game}/cfr_comparison.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # export the measurements to a csv file for plotting later
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
    # allow us to choose which game to benchmark from the terminal
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--game",
        choices=["kuhn", "leduc"],
        default="kuhn"
    )

    args = parser.parse_args()
    run_benchmark(game=args.game)
