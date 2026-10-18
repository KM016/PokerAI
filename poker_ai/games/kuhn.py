# random for distributing cards to players 
import random

# each object will represent a game of Kuhn Poker
class KuhnPoker:
    def __init__(self):
        self.reset()    # resets the game to a fresh state

    def reset(self):
        # deal one card to each player 
        self.cards = random.sample(["J", "Q", "K"], 2)

        # stores whos turn it in - player 0 will act first
        self.current_player = 0

        # record betting actions in order 
        self.history = ""

        # store if the game is ongoing or completed
        self.done = False

    def legal_actions(self):
                # check if the game is done -> if true no more actions are allowed
                if self.done:
                    return []

                # check if the player can bet or check -> if no one has acted or other player checked
                if self.history in ["", "c"]:
                    return ["check", "bet"]

                # check if the player must call or fold -> if bet or check followed by bet
                if self.history in ["b", "cb"]:
                    return ["call", "fold"]

                # for anything else no legal actions
                return []

    def step(self, action):
        # check if the chosen action is legal
        if action not in self.legal_actions():
            raise ValueError(f"Illegal action: {action}")

        # convert actions into symbols
        symbols = {
            "check": "c",
            "bet": "b",
            "call": "k",
            "fold": "f"
        }

        # record the chosen action
        self.history += symbols[action]

        # check if the hand has finished
        if self.history in ["cc", "bk", "bf", "cbk", "cbf"]:
            self.done = True
        else:
            # switch to the other player
            self.current_player = 1 - self.current_player

    
    def get_payoff(self):
        # check if the game is still ongoing 
        if not self.done:
            raise ValueError("Game is not finished")

        # player 1 folds -> player 0 wins
        if self.history == "bf":
            return (1, -1)

        # player 0 folds -> player 1 wins
        if self.history == "cbf":
            return (-1, 1)

        # calculate how much the winner earns at showdown
        winnings = 1 if self.history == "cc" else 2

        # assign numerical values so cards can be compared
        ranks = {"J": 1, "Q": 2, "K": 3}

        # player 0 wins if their card has a higher rank
        if ranks[self.cards[0]] > ranks[self.cards[1]]:
            return (winnings, -winnings)

        # else player 1 wins
        return (-winnings, winnings)

    
    def get_information_set(self):
        # check if the game has finished 
        if self.done:
            raise ValueError("Game is already finished")

        # get card belonging to the current player
        card = self.cards[self.current_player]

        # return their card and the public betting history
        return f"{card}|{self.history}"




