import csv
import time
from pathlib import Path

from poker_ai.algorithms.cfr import CFRTrainer
from poker_ai.algorithms.cfr_plus import CFRPlusTrainer
from poker_ai.evaluation.evaluate import exploitability


# investigates which CFR+ modifications contribute to faster convergence
# this is a sequential component analysis rather than a full ablation study
# we add modifications one at a time rather than testing every possible combination
class AblationTrainer(CFRPlusTrainer):
    def __init__(
        self,
        clip_regrets=False,
        linear_averaging=False,
        alternating=False
    ):
        # use the existing CFR+ trainer to initialise regret and strategy storage
        super().__init__(averaging_delay=0)

        # switches to enable or disable the different CFR+ modifications
        self.clip_regrets = clip_regrets
        self.linear_averaging = linear_averaging
        self.alternating = alternating

    def _apply_regret_updates(self, updates):
        # use CFR+ regret clipping if enabled, otherwise use vanilla CFR
        if self.clip_regrets:
            return CFRPlusTrainer._apply_regret_updates(self, updates)

        return CFRTrainer._apply_regret_updates(self, updates)

    def _averaging_weight(self):
        # use linear averaging if enabled, otherwise weight all iterations equally
        if self.linear_averaging:
            return CFRPlusTrainer._averaging_weight(self)

        return CFRTrainer._averaging_weight(self)

    def train(self, iterations):
        # use alternating player updates if enabled, otherwise update both together
        if self.alternating:
            return CFRPlusTrainer.train(self, iterations)

        return CFRTrainer.train(self, iterations)


# compares different combinations of CFR+ modifications against vanilla CFR
def run_experiment(iterations=10_000):
    # each configuration adds another CFR+ feature to the previous version
    # order: regret clipping -> linear averaging -> alternating updates
    configurations = [
        ("Vanilla CFR", False, False, False),
        ("Clipping only", True, False, False),
        ("Clipping + averaging", True, True, False),
        ("Full CFR+", True, True, True)
    ]

    results = []

    for name, clipping, averaging, alternating in configurations:
        # create a new trainer using the selected modifications
        trainer = AblationTrainer(
            clip_regrets=clipping,
            linear_averaging=averaging,
            alternating=alternating
        )

        # train each configuration for the same number of iterations
        # measure the time taken so we can compare computational cost
        start = time.perf_counter()
        trainer.train(iterations)
        elapsed = time.perf_counter() - start

        # calculate how close the learned strategy is to Nash equilibrium
        score = exploitability(trainer.average_strategy())

        # store the results for each configuration
        results.append({
            "configuration": name,
            "iterations": iterations,
            "exploitability": score,
            "training_seconds": elapsed
        })

        print(f"\n{name}")
        print(f"Exploitability: {score:.8f}")
        print(f"Training time: {elapsed:.3f}s")

    # create the results folder if it does not already exist
    output = Path("results/kuhn/cfr_ablation.csv")
    output.parent.mkdir(parents=True, exist_ok=True)

    # save the results so we can compare the effect of each modification
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "configuration",
                "iterations",
                "exploitability",
                "training_seconds"
            ]
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {output}")

    return results


if __name__ == "__main__":
    run_experiment()
