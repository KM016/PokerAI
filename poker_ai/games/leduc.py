import random
from collections import Counter


# each object represents a game of leduc poker between two players
# leduc uses 6 cards, 2 betting rounds and one shared public card
class LeducPoker:

    # the deck contains 2 copies of each rank
    DECK = ("J", "J", "Q", "Q", "K", "K")

    # numerical values used to compare cards at showdown
    RANKS = {"J": 1, "Q": 2, "K": 3}

    # the first round uses bets of 2 chips and the second uses 4 chips
    BET_SIZES = (2, 4)

    # allow one raise after an opening bet in each betting round
    MAX_RAISES = 1

    def __init__(self, seed=None, deal=None):
        # create a random number generator for dealing cards
        # a seed allows us to reproduce the same random deals during testing
        self.rng = random.Random(seed)

        # initialise a new game with either a random or specified deal
        self.reset(deal)

    def reset(self, deal=None):
        # deal 3 cards from the deck without replacement
        # one card goes to each player and the third becomes the public card
        # we can also provide a fixed deal for testing specific situations
        if deal is None:
            deal = self.rng.sample(self.DECK, 3)

        # convert the deal into a tuple so we can validate its contents
        deal = tuple(deal)

        # check that the deal contains exactly 3 valid cards
        # each rank appears twice in the deck so we cannot deal 3 of the same rank
        # invalid deals raise an error instead of creating an impossible game
        if (
            len(deal) != 3
            or any(card not in self.RANKS for card in deal)
            or any(count > 2 for count in Counter(deal).values())
        ):
            raise ValueError("Invalid Leduc card deal")

        # give the first 2 cards to player 0 and player 1
        self.cards = list(deal[:2])

        # store the third card separately until it is revealed
        # players must not see the public card during the first betting round
        self._future_public_card = deal[2]
        self.public_card = None

        # track which betting round we are currently in
        # round 0 is before the public card is revealed
        # round 1 is after the public card is revealed
        self.round_index = 0

        # player 0 always acts first at the beginning of each round
        self.current_player = 0

        # record every action taken during the game
        # example: cc/bk means both players checked in round 1
        # then one player bet and the other called in round 2
        # the slash separates the two betting rounds
        self.history = ""

        # record actions taken in the current betting round only
        # this resets when we move from round 1 to round 2
        self.round_history = ""

        # track the total chips each player has committed to the pot
        # both players start by paying an ante of 1 chip
        # these contributions are used to calculate the final profit or loss
        self.contributions = [1, 1]

        # track how many chips each player has committed in the current round
        # this lets us calculate how much a player needs to call
        # unlike total contributions these values reset after each round
        self.round_contributions = [0, 0]

        # track how many raises have occurred during the current round
        # this prevents the players from raising indefinitely
        self.raises = 0

        # the game starts unfinished with no winner
        # winner will be 0 or 1 when somebody wins
        # it can also be None if the hand ends in a tie
        self.done = False
        self.winner = None

    @property
    def pot(self):
        # the pot is the sum of all chips committed by both players
        # this automatically includes antes, bets, raises and calls
        return sum(self.contributions)

    def legal_actions(self):
        # if the game has finished then neither player can act
        if self.done:
            return []

        # identify whose turn it is
        player = self.current_player

        # work out how many chips this player needs to match the current bet
        # if both players have contributed equally there is nothing to call
        # example: player 0 contributes 2 and player 1 contributes 0
        # player 1 must add 2 chips to match the bet
        to_call = max(self.round_contributions) - self.round_contributions[player]

        # if there is no outstanding bet the player can check or bet
        # checking adds no chips while betting opens the betting round
        if to_call == 0:
            return ["check", "bet"]

        # if the opponent has bet the player must respond
        # calling matches the bet while folding gives up the hand
        actions = ["call", "fold"]

        # raising is also allowed if the maximum number of raises has not been reached
        # once a raise has occurred the next player can only call or fold
        if self.raises < self.MAX_RAISES:
            actions.append("raise")

        # return all actions currently available to the player
        return actions

    def _contribute(self, player, amount):
        # add the chips paid by a player to their total contribution
        # this increases the size of the pot
        self.contributions[player] += amount

        # also record the amount committed during this betting round
        # we need this to know how much the opponent must call
        self.round_contributions[player] += amount

    def step(self, action):
        # check whether the chosen action is currently allowed
        # illegal moves raise an error and do not change the game
        if action not in self.legal_actions():
            raise ValueError(f"Illegal action: {action}")

        # identify the player taking the action and their opponent
        player = self.current_player
        opponent = 1 - player

        # select the fixed bet size for the current betting round
        # round 0 uses 2 chips and round 1 uses 4 chips
        bet_size = self.BET_SIZES[self.round_index]

        # convert action names into shorter symbols for the betting history
        # these make information sets easier to store and compare
        symbols = {
            "check": "c",
            "bet": "b",
            "call": "k",
            "fold": "f",
            "raise": "r"
        }

        # record the action in the full game history
        # also record it separately for the current betting round
        self.history += symbols[action]
        self.round_history += symbols[action]

        if action == "check":
            # checking means the player does not add anything to the pot
            # if both players check then the betting round is finished
            if self.round_history == "cc":
                self._finish_round()

            # otherwise the first player has checked and the opponent must respond
            # the opponent can either check back or place a bet
            else:
                self.current_player = opponent

        elif action == "bet":
            # betting adds the fixed bet size to the pot
            # example: in round 0 the player adds 2 chips
            self._contribute(player, bet_size)

            # after a bet the opponent must decide whether to call, fold or raise
            self.current_player = opponent

        elif action == "raise":
            # a raise must first match the opponents outstanding bet
            # then add one more fixed bet increment on top
            # example: facing a bet of 2 chips, raising costs 4 chips in total
            to_call = max(self.round_contributions) - self.round_contributions[player]
            self._contribute(player, to_call + bet_size)

            # record that a raise has occurred in this round
            # because MAX_RAISES is 1, the opponent cannot raise again
            self.raises += 1

            # give the opponent the opportunity to call or fold
            self.current_player = opponent

        elif action == "call":
            # calling means matching the amount already committed by the opponent
            # example: if the opponent has bet 2 chips we add 2 chips to match
            to_call = max(self.round_contributions) - self.round_contributions[player]
            self._contribute(player, to_call)

            # once a bet or raise is called the betting round is finished
            # we either reveal the public card or move to showdown
            self._finish_round()

        elif action == "fold":
            # folding immediately ends the game
            # the opponent wins the entire pot without needing a showdown
            self.done = True
            self.winner = opponent

    def _finish_round(self):
        # this function is called when a betting round finishes
        # this happens after both players check or after somebody calls

        if self.round_index == 0:
            # the first betting round has ended
            # move to the second round and reveal the shared public card
            self.round_index = 1
            self.public_card = self._future_public_card

            # reset the betting contributions for the new round
            # we do not reset total contributions because the pot carries over
            # both players start the second round with no outstanding bet
            self.round_contributions = [0, 0]

            # allow another raise in the second betting round
            self.raises = 0

            # clear the current round history while keeping the full history
            # add a slash to show where the first round ended
            self.round_history = ""
            self.history += "/"

            # player 0 acts first again at the start of the second round
            self.current_player = 0

        else:
            # the second betting round has ended
            # both players are still in the game so we determine the winner
            self.done = True
            self.winner = self._showdown_winner()

    def _showdown_winner(self):
        # compare the hands after both betting rounds have finished
        # each player has one private card and shares the same public card

        # the strongest possible hand is a pair with the public card
        # if neither player has a pair then the higher private card wins
        # represent strength using a tuple containing pair status and card rank
        # python compares tuples from left to right so pairs take priority
        strengths = [
            (
                int(card == self.public_card),
                self.RANKS[card]
            )
            for card in self.cards
        ]

        # if player 0 has the stronger hand then player 0 wins
        if strengths[0] > strengths[1]:
            return 0

        # if player 1 has the stronger hand then player 1 wins
        if strengths[1] > strengths[0]:
            return 1

        # if both hands have equal strength the game ends in a tie
        # this can happen because leduc has two copies of every rank
        return None

    def get_payoff(self):
        # we can only calculate the final profit or loss once the hand is finished
        if not self.done:
            raise ValueError("Game is not finished")

        # if the hand ends in a tie both players receive half the pot
        # their total contributions are equal at showdown
        # this means neither player makes a net profit or loss
        if self.winner is None:
            return (0, 0)

        # identify the player who lost the hand
        loser = 1 - self.winner

        # payoffs represent net profit rather than the entire pot
        # the winner gets back their own chips plus the opponents chips
        # therefore their actual profit is the amount the opponent contributed
        # the loser loses the chips they committed during the hand
        payoffs = [0, 0]
        payoffs[self.winner] = self.contributions[loser]
        payoffs[loser] = -self.contributions[loser]

        # return the profit or loss for player 0 and player 1
        # this is a zero sum game so both payoffs always add up to zero
        return tuple(payoffs)

    def get_information_set(self):
        # information sets represent what the current player knows
        # the agent cannot use hidden information when making decisions

        # a player cannot make decisions once the game has finished
        if self.done:
            raise ValueError("Game is already finished")

        # the player knows their own private card but not the opponents card
        card = self.cards[self.current_player]

        # the public card is hidden in round 0 and visible in round 1
        # use a dash to represent an unrevealed public card
        board = self.public_card if self.public_card is not None else "-"

        # combine the private card, public card and full betting history
        # example: J|-| means holding a jack at the beginning of round 0
        # example: J|K|cc/ means holding a jack after both players checked
        # in round 0 and a king was revealed for round 1
        # agents use this information to determine which strategy to follow
        return f"{card}|{board}|{self.history}"
