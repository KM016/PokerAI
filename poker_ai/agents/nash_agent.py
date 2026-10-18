
import random

# creates an instance of a player who follows the nash equilibrium strategy 
class NashAgent:
    def __init__(self, seed=None):
        # seperate rng for reproducible decisions
        self.rng = random.Random(seed)

        # Nash equilibrium probabilities for each information set
        # each action is assigned a probability

        self.strategy = {
            # player 0: first action and no betting history
            "J|": {"check": 2/3, "bet": 1/3},   # Jack -> weakest card -> bluff 1/3 of the time else call 
            "Q|": {"check": 1, "bet": 0},   # Queen -> check as jack may fold but king will call
            "K|": {"check": 0, "bet": 1},   # King -> strongest card so always bet for value  

            # player 1 when betting history is 'check' -> same as above 
            "J|c": {"check": 2/3, "bet": 1/3},     # bluff 1/3 so a Q may fold
            "Q|c": {"check": 1, "bet": 0},  # check instead of risking bet 
            "K|c": {"check": 0, "bet": 1},  # always bet as you cannot be beat 

            # player 1 when history is 'bet'
            "J|b": {"call": 0, "fold": 1},  # always fold as you cannot beat anything
            # call 1/3 in so p0 doesnt bluff to often -> makes p0 indifferent between checking and bluffing with J
            "Q|b": {"call": 1/3, "fold": 2/3},  
            "K|b": {"call": 1, "fold": 0},  # always call as you will always win 

            # player 0 when they check but player 1 bets 
            "J|cb": {"call": 0, "fold": 1}, # always fold as you cannot win 
            "Q|cb": {"call": 2/3, "fold": 1/3}, # makes player 1 indifferent between bluffing and checking with J
            # King will always win so call but under this strategy p0 with King will always bet anyways
            "K|cb": {"call": 1, "fold": 0},
        }   # lookup table that maps information to probabilities of each action 

    def choose_action(self, information_set, legal_actions):
        # lookup probabilities for the current situation
        if information_set not in self.strategy:
            raise ValueError("Unknown information set")

        probabilities = self.strategy[information_set]

        # male sure the stored actions match the games rules
        if set(probabilities) != set(legal_actions):
            raise ValueError("Legal actions do not match strategy")

        # randomly select an action using its probability
        return self.rng.choices(
            list(probabilities.keys()),
            weights=list(probabilities.values()),
            k=1
        )[0]
