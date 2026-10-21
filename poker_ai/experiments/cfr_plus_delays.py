import time

from poker_ai.algorithms.cfr_plus import CFRPlusTrainer
from poker_ai.evaluation.evaluate import exploitability


# tests how different averaging delays affect CFR+ convergence
def run_experiment():
    # different numbers of early iterations to exclude from the average strategy
    delays = [0, 10, 100, 1000]

    # train each version for the same number of iterations for a fair comparison
    iterations = 10_000

    for delay in delays:
        # create a new CFR+ trainer with the selected averaging delay
        trainer = CFRPlusTrainer(averaging_delay=delay)

        # measure how long training takes
        start = time.perf_counter()
        trainer.train(iterations)
        elapsed = time.perf_counter() - start

        # get the learned average strategy and measure how exploitable it is
        # lower exploitability means the strategy is closer to Nash equilibrium
        strategy = trainer.average_strategy()
        score = exploitability(strategy)

        # print the results for each averaging delay
        print(f"Averaging delay: {delay}")
        print(f"Exploitability: {score:.8f}")
        print(f"Training time: {elapsed:.3f}s")
        print()


if __name__ == "__main__":
    run_experiment()
