
import pytest
from poker_ai.algorithms.cfr_plus import CFRPlusTrainer


def test_cfr_plus_clips_negative_regrets():
    trainer = CFRPlusTrainer()

    # Initialise an information set
    trainer.get_strategy("J|", ["check", "bet"])

    # Apply one positive and one negative update
    trainer._apply_regret_updates({
        "J|": {"check": -4, "bet": 3}
    })

    # Negative regrets should be clipped to zero
    assert trainer.regret_sum["J|"] == {
        "check": 0.0,
        "bet": 3.0
    }


def test_cfr_plus_repeated_updates():
    trainer = CFRPlusTrainer()
    trainer.get_strategy("J|", ["check", "bet"])

    trainer._apply_regret_updates({
        "J|": {"check": 5, "bet": 3}
    })

    trainer._apply_regret_updates({
        "J|": {"check": -8, "bet": 2}
    })

    assert trainer.regret_sum["J|"] == {
        "check": 0.0,
        "bet": 5.0
    }


def test_delayed_averaging():
    trainer = CFRPlusTrainer(averaging_delay=2)

    # First two iterations don't contribute to averaging
    trainer.train(2)

    assert sum(trainer.strategy_sum["J|"].values()) == pytest.approx(0)

    # Third iteration has weight 1
    trainer.train(1)

    assert trainer.iteration == 3

    # J| occurs in 2 of 6 equally likely deals
    assert sum(trainer.strategy_sum["J|"].values()) == pytest.approx(1 / 3)


def test_linear_averaging():
    trainer = CFRPlusTrainer(averaging_delay=0)

    trainer.train(2)

    # Iteration weights: 1 and 2
    # J| occurs with probability 2/6
    # Total weight: (1 + 2) * (2/6) = 1
    total = sum(trainer.strategy_sum["J|"].values())

    assert total == pytest.approx(1.0)


# TEST 5: Check that players are updated separately and in order
def test_alternating_player_updates():

    class TrackingTrainer(CFRPlusTrainer):
        def __init__(self):
            super().__init__()
            self.updated_infos = []

        def _apply_regret_updates(self, regret_updates):
            # Record which information sets were updated
            self.updated_infos.append(set(regret_updates))

            super()._apply_regret_updates(regret_updates)

    trainer = TrackingTrainer()
    trainer.train(1)

    cards = ["J", "Q", "K"]

    # Player 0 acts at histories "" and "cb"
    player0_infos = {
        f"{card}|{history}"
        for card in cards
        for history in ("", "cb")
    }

    # Player 1 acts at histories "c" and "b"
    player1_infos = {
        f"{card}|{history}"
        for card in cards
        for history in ("c", "b")
    }

    assert trainer.updated_infos == [
        player0_infos,
        player1_infos
    ]


# TEST 6: Strategy averaging must not count twice
def test_alternating_average_not_double_counted():
    trainer = CFRPlusTrainer(averaging_delay=0)

    trainer.train(1)

    assert trainer.iteration == 1

    # Each of these information sets occurs in 2 of 6 deals
    # With iteration weight 1, accumulated weight = 1/3
    assert sum(trainer.strategy_sum["J|"].values()) == pytest.approx(1 / 3)
    assert sum(trainer.strategy_sum["J|c"].values()) == pytest.approx(1 / 3)



from poker_ai.experiments.cfr_ablation import AblationTrainer
from poker_ai.algorithms.cfr import CFRTrainer
from poker_ai.evaluation.evaluate import exploitability



def test_ablation_matches_vanilla_cfr():
    standard = CFRTrainer()
    ablation = AblationTrainer()

    standard.train(100)
    ablation.train(100)

    # Compare the strategies at every information set
    standard_strategy = standard.average_strategy()
    ablation_strategy = ablation.average_strategy()

    assert standard_strategy.keys() == ablation_strategy.keys()

    for info in standard_strategy:
        assert ablation_strategy[info] == pytest.approx(
            standard_strategy[info]
        )

