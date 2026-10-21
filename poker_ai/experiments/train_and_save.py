from pathlib import Path

from poker_ai.algorithms.cfr import CFRTrainer
from poker_ai.algorithms.cfr_plus import CFRPlusTrainer
from poker_ai.agents.strategy_io import save_strategy, load_strategy
from poker_ai.evaluation.evaluate import exploitability


# trains and saves our CFR and CFR+ strategies for future use
def train_and_save(iterations=10_000):
    # create a separate trainer for each algorithm
    trainers = {
        "cfr": CFRTrainer(),
        "cfr_plus": CFRPlusTrainer(averaging_delay=0)
    }

    for name, trainer in trainers.items():
        # train the algorithm and extract its average strategy
        trainer.train(iterations)
        strategy = trainer.average_strategy()

        # save the strategy so we do not need to retrain it every time
        path = Path(f"models/kuhn/{name}.json")
        save_strategy(strategy, path)

        # load the strategy again to check that nothing has changed
        loaded = load_strategy(path)
        assert loaded == strategy

        # measure how close the saved strategy is to Nash equilibrium
        score = exploitability(loaded)

        print(f"\nModel: {name}")
        print(f"Iterations: {iterations:,}")
        print(f"Exploitability: {score:.8f}")
        print(f"Saved to: {path}")


if __name__ == "__main__":
    train_and_save()
