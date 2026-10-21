import json

import pytest
import torch

from poker_ai.agents.dqn_agent import DQNAgent
from poker_ai.agents.strategy_io import load_strategy, save_strategy
from poker_ai.algorithms.dqn import (
    LEDUC_INPUT_SIZE,
    LeducDQNNetwork,
    encode_leduc_info_set
)
from poker_ai.evaluation.leduc_evaluate import (
    baseline_strategy,
    information_sets
)
from poker_ai.experiments.train_dqn import (
    greedy_strategy,
    run_experiment
)


# check that leduc states produce the expected number of input features
def test_leduc_encoding():
    before = encode_leduc_info_set("J|-|")
    after = encode_leduc_info_set("J|K|cc/")

    assert before.shape == (LEDUC_INPUT_SIZE,)
    assert after.shape == (LEDUC_INPUT_SIZE,)
    assert not torch.equal(before, after)


# different private cards and public histories must produce different inputs
def test_leduc_states_are_distinct():
    states = [
        "J|-|",
        "Q|-|",
        "J|-|c",
        "J|K|cc/",
        "J|K|cc/b"
    ]

    encoded = [encode_leduc_info_set(info) for info in states]

    for i in range(len(encoded)):
        for j in range(i + 1, len(encoded)):
            assert not torch.equal(encoded[i], encoded[j])


# the network must predict one q value for each of the five actions
def test_leduc_network():
    network = LeducDQNNetwork()
    state = encode_leduc_info_set("K|Q|cc/")

    output = network(state)

    assert output.shape == (5,)
    assert torch.isfinite(output).all()


# illegal actions must never be selected even with high predicted q values
def test_leduc_action_masking():
    agent = DQNAgent(game="leduc", epsilon=0, seed=1)

    with torch.no_grad():
        for parameter in agent.policy_net.parameters():
            parameter.zero_()

        # raise receives the highest q value but may be illegal
        agent.policy_net.network[-1].bias.copy_(
            torch.tensor([0.0, 0.0, 2.0, 1.0, 100.0])
        )

    # a raise is legal when first facing an opening bet
    assert agent.choose_action(
        "J|-|b", ["call", "fold", "raise"]
    ) == "raise"

    # after a raise the only options are call and fold
    # the network must ignore the high predicted value for raise
    assert agent.choose_action(
        "J|-|br", ["call", "fold"]
    ) == "call"


# bellman targets must also exclude illegal future actions
def test_leduc_target_masking():
    agent = DQNAgent(
        game="leduc",
        batch_size=1,
        learning_rate=0.01,
        target_update=100,
        loss="mse",
        seed=1
    )

    with torch.no_grad():
        for parameter in agent.policy_net.parameters():
            parameter.zero_()

        for parameter in agent.target_net.parameters():
            parameter.zero_()

        # the illegal raise has a high value
        # both legal future actions have negative values
        agent.target_net.network[-1].bias.copy_(
            torch.tensor([0.0, 0.0, -2.0, -1.0, 100.0])
        )

    # after the learner bets following a check the opponent raises
    # the next learner decision can only be call or fold
    agent.remember(
        "J|-|c",
        "bet",
        0,
        ["check", "bet"],
        "J|-|cbr",
        ["call", "fold"]
    )

    agent.learn()

    # the target should be negative because raise is not legal
    # without masking the illegal raise would produce a positive target
    assert agent.policy_net.network[-1].bias[1].item() < 0


# convert neural predictions into a complete legal leduc policy
def test_leduc_greedy_strategy():
    agent = DQNAgent(game="leduc", seed=1)

    strategy = greedy_strategy(agent, game_name="leduc")

    assert len(strategy) == len(information_sets())

    for info, _, actions in information_sets():
        assert set(strategy[info]) == set(actions)
        assert sum(strategy[info].values()) == pytest.approx(1)


# verify that short training saves reusable model files
def test_leduc_dqn_training(tmp_path):
    model_dir = tmp_path / "models"
    output_dir = tmp_path / "results"

    # use a temporary reference strategy for this isolated test
    save_strategy(
        baseline_strategy("Random"),
        model_dir / "cfr_plus.json"
    )

    results = run_experiment(
        game="leduc",
        checkpoints=(16, 32),
        opponent_name="Random",
        loss="mse",
        seed=101,
        batch_size=4,
        learn_every=2,
        model_dir=model_dir,
        output_dir=output_dir
    )

    filename = "dqn_random_mse"
    policy = load_strategy(model_dir / f"{filename}.json")

    assert len(results) == 2
    assert results[-1]["learning_updates"] > 0
    assert len(policy) == len(information_sets())

    assert (model_dir / f"{filename}.pt").exists()
    assert (output_dir / f"{filename}.csv").exists()

    metadata = json.loads(
        (model_dir / f"{filename}_metadata.json").read_text()
    )

    assert metadata["game"] == "leduc"
    assert metadata["loss"] == "mse"
    assert metadata["network_architecture"] == [55, 32, 32, 5]
