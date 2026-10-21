import pytest
import torch

from poker_ai.agents.dqn_agent import DQNAgent


def test_initial_target_network_matches_policy():
    agent = DQNAgent(seed=42)
    for policy, target in zip(agent.policy_net.parameters(), agent.target_net.parameters()):
        assert torch.equal(policy, target)
        assert not target.requires_grad


def test_initial_network_is_reproducible():
    a = DQNAgent(seed=42)
    b = DQNAgent(seed=42)
    for first, second in zip(a.policy_net.parameters(), b.policy_net.parameters()):
        assert torch.equal(first, second)


def test_greedy_action_uses_network_predictions():
    agent = DQNAgent(epsilon=0, seed=1)
    with torch.no_grad():
        for parameter in agent.policy_net.parameters():
            parameter.zero_()
        agent.policy_net.network[-1].bias.copy_(torch.tensor([-1.0, 2.0]))

    assert agent.choose_action("J|", ["check", "bet"]) == "bet"
    assert agent.choose_action("K|b", ["call", "fold"]) == "fold"
    assert agent.choose_action("K|b", ["fold", "call"]) == "fold"


def test_epsilon_explores_both_actions():
    agent = DQNAgent(epsilon=1, seed=3)
    choices = {agent.choose_action("J|", ["check", "bet"]) for _ in range(100)}
    assert choices == {"check", "bet"}


def test_replay_memory_has_fixed_capacity():
    agent = DQNAgent(replay_capacity=2, seed=3)
    for reward in range(3):
        agent.remember("J|", "bet", reward, ["check", "bet"])
    assert len(agent.memory) == 2
    assert [transition[2] for transition in agent.memory] == [1.0, 2.0]


def test_learning_waits_for_full_batch():
    agent = DQNAgent(batch_size=2, seed=1)
    agent.remember("J|", "bet", 1, ["check", "bet"])
    assert agent.learn() is None
    assert agent.learning_steps == 0


def test_terminal_learning_changes_weights_and_syncs_target():
    agent = DQNAgent(batch_size=1, learning_rate=0.01, target_update=1, seed=1)
    with torch.no_grad():
        for parameter in agent.policy_net.parameters():
            parameter.zero_()
    agent.remember("J|", "bet", 1, ["check", "bet"])
    loss = agent.learn()
    assert loss is not None and loss > 0
    assert agent.learning_steps == 1
    with torch.no_grad():
        assert agent.policy_net( torch.tensor([1., 0., 0., 1., 0., 0., 0.]))[1] > 0
    for policy, target in zip(agent.policy_net.parameters(), agent.target_net.parameters()):
        assert torch.equal(policy, target)


def test_next_state_bootstraps_from_target_network():
    agent = DQNAgent(batch_size=1, gamma=1, learning_rate=0.01, target_update=100, seed=1)
    with torch.no_grad():
        for parameter in agent.policy_net.parameters():
            parameter.zero_()
        for parameter in agent.target_net.parameters():
            parameter.zero_()
        agent.target_net.network[-1].bias.copy_(torch.tensor([2.0, 0.0]))
    agent.remember("J|", "check", 0, ["check", "bet"], "J|cb", ["call", "fold"])
    agent.learn()
    with torch.no_grad():
        assert agent.policy_net(torch.tensor([1., 0., 0., 1., 0., 0., 0.]))[0] > 0
        assert agent.target_net.network[-1].bias[0] == 2.0


def test_remember_rejects_invalid_actions():
    agent = DQNAgent(seed=1)
    with pytest.raises(ValueError):
        agent.remember("J|", "call", 1, ["check", "bet"])
    with pytest.raises(ValueError):
        agent.remember("J|", "check", 0, ["check", "bet"], "J|cb")

# choose the loss explicitly so previous and improved training setups remain available
def test_dqn_supports_huber_and_mse():
    assert isinstance(DQNAgent(loss="huber", seed=7).loss_function, torch.nn.SmoothL1Loss)
    assert isinstance(DQNAgent(loss="mse", seed=7).loss_function, torch.nn.MSELoss)


# both losses must penalise an error but they weight large errors differently
def test_mse_penalises_large_errors_more():
    actual = torch.tensor([2.0])
    target = torch.tensor([-2.0])
    squared = DQNAgent(loss="mse", seed=1).loss_function(actual, target)
    huber = DQNAgent(loss="huber", seed=1).loss_function(actual, target)
    assert squared.item() == pytest.approx(16.0)
    assert huber.item() == pytest.approx(3.5)


def test_dqn_rejects_unknown_loss():
    with pytest.raises(ValueError, match="loss"):
        DQNAgent(loss="unknown")
