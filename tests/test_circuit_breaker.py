"""Tests for the embedding provider circuit breaker."""

from dimensionalbase.embeddings.circuit_breaker import CircuitBreaker, CircuitState


class TestCircuitBreakerStates:
    def test_initial_state_is_closed(self):
        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED
        assert not cb.is_open

    def test_failures_below_threshold_stay_closed(self):
        cb = CircuitBreaker(failure_threshold=5)
        for _ in range(4):
            cb.record_failure()
        assert cb.state == CircuitState.CLOSED

    def test_reaching_threshold_opens_circuit(self):
        cb = CircuitBreaker(failure_threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.is_open

    def test_success_after_failures_resets(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb.state == CircuitState.CLOSED
        assert cb._consecutive_failures == 0

    def test_success_closes_open_circuit(self):
        cb = CircuitBreaker(failure_threshold=1)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_reset(self):
        cb = CircuitBreaker(failure_threshold=1)
        cb.record_failure()
        assert cb.is_open
        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb._consecutive_failures == 0


class TestCircuitBreakerRecovery:
    def test_half_open_after_recovery_timeout(self):
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.0)
        cb.record_failure()
        # With 0 recovery timeout, accessing .state immediately transitions
        # to HALF_OPEN (the recovery window has already passed)
        import time
        time.sleep(0.01)
        assert cb.state == CircuitState.HALF_OPEN

    def test_dropped_event_counter(self):
        from dimensionalbase.events.bus import EventBus
        from dimensionalbase.core.types import Event, EventType

        bus = EventBus(max_history=3)
        for i in range(5):
            bus.emit(Event(type=EventType.CHANGE, path=f"test/{i}"))
        assert bus.dropped_event_count > 0
        assert len(bus.get_history()) == 3
