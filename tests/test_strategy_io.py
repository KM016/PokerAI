from poker_ai.agents.strategy_io import save_strategy, load_strategy
from poker_ai.agents.strategy_agent import StrategyAgent
from poker_ai.algorithms.cfr_plus import CFRPlusTrainer


def test_strategy_save_load_round_trip(tmp_path):
    # use a temporary folder so the test does not modify our real saved strategies
    strategy = {
        "J|": {"check": 0.3, "bet": 0.7},
        "K|b": {"call": 1.0, "fold": 0.0}
    }
    path = tmp_path / "models" / "kuhn" / "test_strategy.json"

    save_strategy(strategy, path)
    loaded = load_strategy(path)

    assert path.exists()
    assert loaded == strategy


def test_loaded_cfr_plus_strategy_works_with_agent(tmp_path):
    # train a CFR+ strategy and check that we can reuse it after loading
    trainer = CFRPlusTrainer()
    trainer.train(100)
    strategy = trainer.average_strategy()
    path = tmp_path / "cfr_plus.json"

    save_strategy(strategy, path)
    loaded = load_strategy(path)
    agent = StrategyAgent(loaded, seed=42)

    assert loaded == strategy
    assert agent.choose_action("J|", ["check", "bet"]) in ["check", "bet"]
