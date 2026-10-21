import json
from pathlib import Path


# saves a learned strategy as a JSON file so we can reuse it without retraining
def save_strategy(strategy, path):
    # create the folder if it does not already exist
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # save the information sets and action probabilities
    with path.open("w", encoding="utf-8") as file:
        json.dump(strategy, file, indent=4, allow_nan=False)


# loads a previously saved strategy from a JSON file
def load_strategy(path):
    # read the stored information sets and action probabilities
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)
