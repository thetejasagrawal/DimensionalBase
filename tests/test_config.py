"""Tests for DimensionalBaseConfig propagation."""

from dimensionalbase import DimensionalBase, DimensionalBaseConfig


class TestConfigDefaults:
    """Default config matches prior hard-coded behaviour."""

    def test_default_config_instantiation(self):
        cfg = DimensionalBaseConfig()
        assert cfg.contradiction_threshold == 0.75
        assert cfg.staleness_seconds == 3600.0
        assert cfg.default_trust == 0.5
        assert cfg.event_history_max == 1000
        assert cfg.pagerank_epsilon == 1e-6

    def test_db_accepts_config(self):
        cfg = DimensionalBaseConfig(staleness_seconds=999)
        db = DimensionalBase(config=cfg)
        assert db._config.staleness_seconds == 999
        db.close()

    def test_db_default_config(self):
        db = DimensionalBase()
        assert db._config is not None
        assert db._config.contradiction_threshold == 0.75
        db.close()


class TestConfigPropagation:
    """Config values reach the subsystems that use them."""

    def test_event_bus_max_history(self):
        cfg = DimensionalBaseConfig(event_history_max=5)
        db = DimensionalBase(config=cfg)
        assert db._event_bus._max_history == 5
        db.close()

    def test_trust_engine_receives_k_factor(self):
        cfg = DimensionalBaseConfig(trust_k_factor=64.0)
        db = DimensionalBase(config=cfg)
        assert db._trust._k_factor == 64.0
        db.close()

    def test_trust_engine_receives_default_trust(self):
        cfg = DimensionalBaseConfig(default_trust=0.8)
        db = DimensionalBase(config=cfg)
        profile = db._trust.get_or_create_profile("test-agent")
        assert profile.global_trust == 0.8
        db.close()

    def test_confidence_engine_receives_weights(self):
        cfg = DimensionalBaseConfig(confirmation_weight=3.0, contradiction_weight=5.0)
        db = DimensionalBase(config=cfg)
        assert db._confidence._confirmation_weight == 3.0
        assert db._confidence._contradiction_weight == 5.0
        db.close()

    def test_reasoning_receives_thresholds(self):
        cfg = DimensionalBaseConfig(
            staleness_seconds=999, contradiction_threshold=0.9, summary_threshold=20,
        )
        db = DimensionalBase(config=cfg)
        assert db._reasoning._staleness_threshold == 999
        assert db._reasoning._contradiction_threshold == 0.9
        assert db._reasoning._summary_threshold == 20
        db.close()

    def test_pagerank_convergence_params(self):
        cfg = DimensionalBaseConfig(pagerank_max_iterations=5, pagerank_epsilon=0.1)
        db = DimensionalBase(config=cfg)
        assert db._trust._pagerank_max_iterations == 5
        assert db._trust._pagerank_epsilon == 0.1
        db.close()
