import torch
from torch import nn


CARDS = ("J", "Q", "K")
HISTORIES = ("", "c", "b", "cb")


# converts a poker information set into numerical inputs for our neural network
def encode_info_set(info_set):
    # split the information set into the private card and public betting history
    # example: J|cb means holding a jack after a check followed by a bet
    parts = info_set.split("|")

    if len(parts) != 2:
        raise ValueError("Invalid information set")

    card, history = parts

    if card not in CARDS or history not in HISTORIES:
        raise ValueError("Invalid card or betting history")

    # create seven inputs, initially all set to zero
    # first 3 represent cards and remaining 4 represent betting histories
    features = torch.zeros(7, dtype=torch.float32)

    # use one hot encoding to identify the private card
    # example: J becomes [1, 0, 0]
    features[CARDS.index(card)] = 1.0

    # use another one hot encoding to identify the betting history
    # example: cb becomes [0, 0, 0, 1]
    features[3 + HISTORIES.index(history)] = 1.0

    # the full input contains exactly two values equal to 1
    return features


# neural network that estimates the q values of the two legal actions
class DQNNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        # input layer takes the seven encoded poker features
        # hidden layers learn relationships between cards and betting histories
        # output layer predicts one q value for each legal action
        self.network = nn.Sequential(
            nn.Linear(7, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, state):
        # pass the encoded information set through the network
        # output contains two estimated action values rather than probabilities
        return self.network(state)
