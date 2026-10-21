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


# fixed action ordering for the leduc neural network
# each output neuron predicts the q value of one possible action
LEDUC_ACTIONS = ("check", "bet", "call", "fold", "raise")

# symbols used to represent individual actions in the betting history
# the dash represents an unused position in the history
LEDUC_SYMBOLS = ("-", "c", "b", "k", "f", "r")

# 3 private card features + 4 public card features
# each of the 2 rounds has up to 4 actions with 6 encoding options
LEDUC_INPUT_SIZE = 3 + 4 + 2 * 4 * 6


# convert a leduc information set into numerical neural network inputs
def encode_leduc_info_set(info_set):
    # split the information set into the private card, public card and history
    # example: J|K|cc/b means a jack with a king on the board
    # the first round ended with two checks and round 2 contains a bet
    parts = info_set.split("|")

    if len(parts) != 3:
        raise ValueError("Invalid Leduc information set")

    card, board, history = parts

    # validate the private card and the public card
    # a dash means the public card has not been revealed yet
    if card not in CARDS or board not in ("-", *CARDS):
        raise ValueError("Invalid Leduc cards")

    # separate the first and second betting rounds
    # before the public card is revealed there should be no slash
    # after the reveal the slash separates the two rounds
    if board == "-":
        if "/" in history:
            raise ValueError("Public card and history do not match")
        rounds = (history, "")

    else:
        if history.count("/") != 1:
            raise ValueError("Public card and history do not match")
        rounds = tuple(history.split("/"))

    # each betting round contains at most four actions
    # reject histories containing unknown action symbols
    for round_history in rounds:
        if len(round_history) > 4:
            raise ValueError("Betting history is too long")

        if any(symbol not in LEDUC_SYMBOLS[1:] for symbol in round_history):
            raise ValueError("Invalid betting history")

    # initialise every input feature to zero
    features = torch.zeros(LEDUC_INPUT_SIZE, dtype=torch.float32)

    # first three inputs represent the private card using one hot encoding
    features[CARDS.index(card)] = 1.0

    # next four inputs represent the public card
    # encode an unrevealed card separately from jack, queen and king
    board_options = ("-", *CARDS)
    features[3 + board_options.index(board)] = 1.0

    # encode the public betting actions from both rounds
    # each action position gets six possible features including an empty slot
    # using separate positions preserves the ordering of actions
    for round_index, round_history in enumerate(rounds):
        for position in range(4):

            # fill unused positions with a dash
            symbol = (
                round_history[position]
                if position < len(round_history)
                else "-"
            )

            # find the section of the feature vector for this action position
            slot = round_index * 4 + position
            offset = 7 + slot * len(LEDUC_SYMBOLS)

            # activate the feature corresponding to the observed action
            features[offset + LEDUC_SYMBOLS.index(symbol)] = 1.0

    # the result contains 55 numerical inputs for the neural network
    return features


# leduc network predicts q values for all five possible actions
# it uses the same two hidden layers as our original kuhn dqn
class LeducDQNNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(LEDUC_INPUT_SIZE, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, len(LEDUC_ACTIONS))
        )

    def forward(self, state):
        # output contains q values rather than action probabilities
        # illegal actions will be excluded by the agent
        return self.network(state)

