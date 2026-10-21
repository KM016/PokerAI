import random


# q learning agent that learns the expected value of each action through experience
class QLearningAgent:
    def __init__(self, alpha=0.1, epsilon=0.1, gamma=1.0, seed=None):
        # alpha = learning rate, controls how much new experiences change our estimates
        # epsilon = exploration rate, probability of choosing a random action
        # gamma = discount factor, determines how much future rewards are valued

        # ensure all three parameters are within their valid ranges
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be between 0 and 1")
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be between 0 and 1")
        if not 0 <= gamma <= 1:
            raise ValueError("gamma must be between 0 and 1")

        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma

        # separate rng for reproducible exploration and action selection
        self.rng = random.Random(seed)

        # stores estimated action values for every information set
        # example: {"J|": {"check": 0.2, "bet": -0.5}}
        self.q_table = {}

    def _get_values(self, info_set, legal_actions):
        # lookup the q values for a particular information set
        # if we have never encountered it before, initialise all actions to 0
        if info_set not in self.q_table:
            self.q_table[info_set] = {
                action: 0.0 for action in legal_actions
            }

        values = self.q_table[info_set]

        # make sure the stored actions match the actions allowed by the game
        if set(values) != set(legal_actions):
            raise ValueError("Q-table actions do not match legal actions")

        return values

    def choose_action(self, info_set, legal_actions):
        # retrieve the current estimates for each available action
        values = self._get_values(info_set, legal_actions)

        # epsilon greedy exploration
        # with probability epsilon we choose a random action to explore
        # this allows the agent to discover actions that might be better
        # otherwise we exploit the best action according to our current estimates
        if self.rng.random() < self.epsilon:
            return self.rng.choice(legal_actions)

        # find the highest q value among the available actions
        best_value = max(values.values())

        # find every action with that value
        # multiple actions may have equal values, especially early in training
        best_actions = [
            action for action in legal_actions
            if values[action] == best_value
        ]

        # randomly choose between equally valued actions
        return self.rng.choice(best_actions)

    # updates our estimate of an action using the reward received
    def update(
        self,
        info_set,
        action,
        reward,
        legal_actions,
        next_info_set=None,
        next_legal_actions=None
    ):
        # get the current q values for the information set
        values = self._get_values(info_set, legal_actions)

        # only allow updates for actions that were legal in that situation
        if action not in values:
            raise ValueError("Action is not legal for this information set")

        # calculate the target value we want our q estimate to approach
        # if the hand has ended, the target is simply the final payoff
        # example: winning a called bet gives a reward of +2
        target = reward

        # if the game continues, we also consider the value of future decisions
        if next_info_set is not None:
            if not next_legal_actions:
                raise ValueError("Next legal actions are required")

            # get the q values at the next information set where our agent acts
            next_values = self._get_values(
                next_info_set, next_legal_actions
            )

            # q learning target = reward + gamma * best estimated future value
            # we assume the agent chooses its best action at its next decision
            target += self.gamma * max(next_values.values())

        # q learning update rule:
        # new q = old q + alpha * (target - old q)
        # target - old q is the error in our current estimate
        # alpha controls how much of that error we use to update the value
        # over many hands the estimates adjust based on observed rewards
        values[action] += self.alpha * (target - values[action])
