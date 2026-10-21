
import pytest
import torch

from poker_ai.algorithms.dqn import DQNNetwork, encode_info_set


# check that the private card and betting history are encoded correctly
def test_information_set_encoding():
    state = encode_info_set("J|")

    assert state.tolist() == [1, 0, 0, 1, 0, 0, 0]
    assert state.dtype == torch.float32


# check that different poker situations produce different inputs
def test_different_information_sets():
    a = encode_info_set("J|")
    b = encode_info_set("J|cb")
    c = encode_info_set("K|")

    assert not torch.equal(a, b)
    assert not torch.equal(a, c)


# invalid information sets should be rejected
@pytest.mark.parametrize(
    "info_set",
    ["X|", "J|bbb", "J", "J|c|b"]
)
def test_invalid_information_sets(info_set):
    with pytest.raises(ValueError):
        encode_info_set(info_set)


# network should produce one q value for each available action
def test_network_output_shape():
    network = DQNNetwork()
    state = encode_info_set("K|b")

    output = network(state)

    assert output.shape == (2,)
    assert torch.isfinite(output).all()


# check that the network can process multiple poker situations together
def test_network_batch():
    network = DQNNetwork()

    states = torch.stack([
        encode_info_set("J|"),
        encode_info_set("Q|c"),
        encode_info_set("K|b")
    ])

    assert network(states).shape == (3, 2)


# check that gradients can flow through the network for training
def test_network_gradients():
    network = DQNNetwork()
    state = encode_info_set("J|")

    loss = network(state).sum()
    loss.backward()

    assert all(
        parameter.grad is not None
        for parameter in network.parameters()
    )
