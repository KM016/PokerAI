import copy
import random
from collections import deque

import torch
from torch import nn

from poker_ai.algorithms.dqn import DQNNetwork, encode_info_set


# uses a neural network to estimate q values instead of storing them in a table
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
        loss="huber"
    ):
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

        self.loss = loss
        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update
        self.rng = random.Random(seed)
        self.memory = deque(maxlen=replay_capacity)
        self.learning_steps = 0

        # initialise the network reproducibly without changing the global torch rng
        if seed is None:
            self.policy_net = DQNNetwork()
        else:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed)
                self.policy_net = DQNNetwork()

        # target network starts as a copy and stays fixed between scheduled updates
        self.target_net = copy.deepcopy(self.policy_net)
        self.target_net.eval()
        self.target_net.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        # use squared error when the goal is estimating mean returns from asymmetric outcomes
        # huber remains available to reproduce our earlier dqn experiments
        self.loss_function = nn.MSELoss() if loss == "mse" else nn.SmoothL1Loss()

    def _actions(self, info_set, legal_actions):
        # the two outputs have a fixed meaning based on the betting history
        # output 0 is check or call and output 1 is bet or fold
        encode_info_set(info_set)
        history = info_set.split("|")[1]
        actions = ("check", "bet") if history in ("", "c") else ("call", "fold")

        if set(legal_actions) != set(actions):
            raise ValueError("Actions do not match the information set")
        return actions

    def choose_action(self, info_set, legal_actions):
        actions = self._actions(info_set, legal_actions)

        # epsilon greedy exploration lets the agent try actions it might otherwise ignore
        if self.rng.random() < self.epsilon:
            return self.rng.choice(list(actions))

        # predict both q values without building a gradient graph during gameplay
        with torch.no_grad():
            values = self.policy_net(encode_info_set(info_set))

        # choose the action with the highest predicted reward
        # the output order follows the actions returned above
        return actions[int(torch.argmax(values).item())]

    def remember(
        self,
        info_set,
        action,
        reward,
        legal_actions,
        next_info_set=None,
        next_legal_actions=None
    ):
        actions = self._actions(info_set, legal_actions)
        if action not in actions:
            raise ValueError("Action is not legal for this information set")

        # a non terminal transition must point to the next decision by our agent
        if next_info_set is not None:
            if next_legal_actions is None:
                raise ValueError("Next legal actions are required")
            self._actions(next_info_set, next_legal_actions)
        elif next_legal_actions is not None:
            raise ValueError("Terminal transitions cannot have next legal actions")

        # store experience so later training can sample from multiple previous hands
        # action is stored as 0 or 1 to match the neural network output
        self.memory.append((info_set, actions.index(action), float(reward), next_info_set))

    def learn(self):
        # wait until enough experiences have been collected for a full training batch
        if len(self.memory) < self.batch_size:
            return None

        # replay random past experiences to reduce dependence on their original order
        batch = self.rng.sample(self.memory, self.batch_size)
        states = torch.stack([encode_info_set(row[0]) for row in batch])
        actions = torch.tensor([row[1] for row in batch], dtype=torch.long)
        rewards = torch.tensor([row[2] for row in batch], dtype=torch.float32)

        # predict the q value of the action that was actually chosen in each experience
        predicted = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        targets = rewards.clone()

        # for ongoing hands add the best future q value from the target network
        # terminal hands only use the final reward since there is no next decision
        continuing = [i for i, row in enumerate(batch) if row[3] is not None]
        if continuing:
            next_states = torch.stack([encode_info_set(batch[i][3]) for i in continuing])
            with torch.no_grad():
                future_values = self.target_net(next_states).max(dim=1).values
                targets[continuing] += self.gamma * future_values

        # backpropagation changes the network weights to reduce prediction errors
        loss = self.loss_function(predicted, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # periodically copy the learned weights into the stable target network
        self.learning_steps += 1
        if self.learning_steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()
