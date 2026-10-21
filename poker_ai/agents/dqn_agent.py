
import copy
import random
from collections import deque

import torch
from torch import nn

from poker_ai.algorithms.dqn import (
    DQNNetwork,
    LeducDQNNetwork,
    LEDUC_ACTIONS,
    encode_info_set,
    encode_leduc_info_set
)


# uses a neural network to estimate q values instead of storing them in a table
# supports both kuhn and leduc by selecting the correct network and actions
class DQNAgent:
    def __init__(
        self,
        learning_rate=0.001,
        epsilon=0.1,
        gamma=1.0,
        batch_size=32,
        replay_capacity=10_000,
        target_update=100,
        seed=None,
        loss="huber",
        game="kuhn"
    ):
        # check that all training settings are valid
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be between 0 and 1")
        if not 0 <= gamma <= 1:
            raise ValueError("gamma must be between 0 and 1")
        if batch_size <= 0 or replay_capacity <= 0 or target_update <= 0:
            raise ValueError("batch_size, replay_capacity and target_update must be positive")
        if loss not in ("huber", "mse"):
            raise ValueError("loss must be huber or mse")
        if game not in ("kuhn", "leduc"):
            raise ValueError("Unknown poker game")

        # store the settings used by the agent during training
        self.game = game
        self.loss = loss
        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update

        # use a separate rng for reproducible exploration and replay sampling
        self.rng = random.Random(seed)

        # store previous experiences so we can learn from randomly sampled batches
        self.memory = deque(maxlen=replay_capacity)
        self.learning_steps = 0

        # kuhn predicts 2 q values while leduc predicts 5 q values
        # choose the network architecture before initialising its weights
        network_class = DQNNetwork if game == "kuhn" else LeducDQNNetwork

        # initialise the network using the chosen architecture
        # setting a seed makes the initial weights reproducible
        if seed is None:
            self.policy_net = network_class()
        else:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                self.policy_net = network_class()

        # create a separate target network with identical initial weights
        # this provides more stable estimates of future q values
        self.target_net = copy.deepcopy(self.policy_net)
        self.target_net.eval()
        self.target_net.requires_grad_(False)

        # adam updates the network weights using gradients from the loss
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(),
            lr=learning_rate
        )

        # choose between squared error and huber loss
        self.loss_function = (
            nn.MSELoss() if loss == "mse" else nn.SmoothL1Loss()
        )

    # convert an information set into numerical network inputs
    def _encode(self, info_set):
        # kuhn uses 7 inputs while leduc uses 55 inputs
        if self.game == "kuhn":
            return encode_info_set(info_set)

        return encode_leduc_info_set(info_set)

    # determine which actions are legal at a particular information set
    def _actions(self, info_set, legal_actions):
        # check that the information set has the correct structure
        self._encode(info_set)

        if self.game == "kuhn":
            # kuhn uses two legal actions at each information set
            history = info_set.split("|")[1]

            actions = (
                ("check", "bet")
                if history in ("", "c")
                else ("call", "fold")
            )

        else:
            # leduc has two betting rounds separated by a slash
            # only the current rounds history determines the legal actions
            history = info_set.split("|")[2].split("/")[-1]

            # when nobody has bet we can check or open betting
            if history in ("", "c"):
                actions = ("check", "bet")

            # facing the opening bet allows calling, folding or raising
            elif history in ("b", "cb"):
                actions = ("call", "fold", "raise")

            # after a raise the opponent can only call or fold
            elif history in ("br", "cbr"):
                actions = ("call", "fold")

            else:
                raise ValueError("Invalid Leduc decision history")

        # ensure that the provided legal actions match the game rules
        if set(legal_actions) != set(actions):
            raise ValueError("Actions do not match the information set")

        return actions

    # map action names onto neural network output positions
    def _action_index(self, action, actions):
        if self.game == "kuhn":
            # kuhn uses two context dependent outputs
            return actions.index(action)

        # leduc always uses check, bet, call, fold, raise in this order
        return LEDUC_ACTIONS.index(action)

    # select an action using epsilon greedy exploration
    def choose_action(self, info_set, legal_actions):
        # identify the legal actions before making a decision
        actions = self._actions(info_set, legal_actions)

        # explore by choosing a random legal action with probability epsilon
        if self.rng.random() < self.epsilon:
            return self.rng.choice(list(actions))

        # predict q values without recording gradients during gameplay
        with torch.no_grad():
            values = self.policy_net(self._encode(info_set))

        # find which output neuron corresponds to each legal action
        indices = [
            self._action_index(action, actions)
            for action in actions
        ]

        # compare only legal actions when choosing the highest q value
        # this prevents the network from selecting illegal actions
        best = max(
            range(len(actions)),
            key=lambda i: values[indices[i]].item()
        )

        return actions[best]

    # store an experience in replay memory for later training
    def remember(
        self,
        info_set,
        action,
        reward,
        legal_actions,
        next_info_set=None,
        next_legal_actions=None
    ):
        # check that the action was legal at the current information set
        actions = self._actions(info_set, legal_actions)

        if action not in actions:
            raise ValueError("Action is not legal for this information set")

        # nonterminal experiences lead to another learner decision
        # we also need the legal actions at that future information set
        if next_info_set is not None:
            if next_legal_actions is None:
                raise ValueError("Next legal actions are required")

            next_actions = self._actions(
                next_info_set,
                next_legal_actions
            )

        else:
            # terminal experiences have no future decisions
            if next_legal_actions is not None:
                raise ValueError("Terminal transitions cannot have next legal actions")

            next_actions = None

        # store the information set, action index, reward and next state
        # next_actions is needed to mask illegal outputs during learning
        self.memory.append((
            info_set,
            self._action_index(action, actions),
            float(reward),
            next_info_set,
            next_actions
        ))

    # learn from a randomly sampled batch of previous experiences
    def learn(self):
        # wait until enough experiences exist to construct a full batch
        if len(self.memory) < self.batch_size:
            return None

        # sample previous decisions instead of only using recent experiences
        batch = self.rng.sample(self.memory, self.batch_size)

        # convert information sets into numerical feature vectors
        states = torch.stack([
            self._encode(row[0]) for row in batch
        ])

        # extract action indices and observed rewards
        actions = torch.tensor(
            [row[1] for row in batch],
            dtype=torch.long
        )
        rewards = torch.tensor(
            [row[2] for row in batch],
            dtype=torch.float32
        )

        # predict q values for the actions the learner actually selected
        predicted = (
            self.policy_net(states)
            .gather(1, actions.unsqueeze(1))
            .squeeze(1)
        )

        # terminal transitions use the observed reward as their target
        targets = rewards.clone()

        # identify transitions where the learner gets another decision
        continuing = [
            i for i, row in enumerate(batch)
            if row[3] is not None
        ]

        if continuing:
            # encode the next information set for each ongoing experience
            next_states = torch.stack([
                self._encode(batch[i][3])
                for i in continuing
            ])

            # create a mask identifying legal network outputs
            # kuhn has 2 outputs while leduc has 5
            output_count = (
                2 if self.game == "kuhn" else len(LEDUC_ACTIONS)
            )

            mask = torch.zeros(
                (len(continuing), output_count),
                dtype=torch.bool
            )

            # mark every legal future action as true in the mask
            for row_index, batch_index in enumerate(continuing):
                next_actions = batch[batch_index][4]

                for action in next_actions:
                    index = self._action_index(action, next_actions)
                    mask[row_index, index] = True

            # estimate future q values using the target network
            with torch.no_grad():
                future_values = self.target_net(next_states)

                # replace illegal action values with negative infinity
                # this prevents them from affecting the maximum future value
                future_values = future_values.masked_fill(
                    ~mask,
                    float("-inf")
                )

                best_future = future_values.max(dim=1).values

                # bellman target = reward + discounted best future q value
                targets[continuing] += self.gamma * best_future

        # calculate the difference between predictions and target values
        loss = self.loss_function(predicted, targets)

        # backpropagate the error and update the policy network weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # periodically copy the policy network into the target network
        self.learning_steps += 1

        if self.learning_steps % self.target_update == 0:
            self.target_net.load_state_dict(
                self.policy_net.state_dict()
            )

        return loss.item()
