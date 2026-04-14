"""
Circuit breaker for embedding providers.

When the embedding API goes down, avoid paying the latency cost of repeated
failing requests.  Instead, open the circuit and fall back to text-only mode
until the provider recovers.

States:
    CLOSED     Normal — calls go through.
    OPEN       Failing — calls are skipped (returns None immediately).
    HALF_OPEN  Recovery probe — one call is allowed through to test.
"""

from __future__ import annotations

import logging
import threading
import time
from enum import Enum
from typing import Optional

logger = logging.getLogger("dimensionalbase.embeddings.circuit_breaker")


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Simple circuit breaker for embedding provider calls."""

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
    ):
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._last_failure_time: float = 0.0
        self._lock = threading.Lock()

    @property
    def state(self) -> CircuitState:
        with self._lock:
            if self._state == CircuitState.OPEN:
                if time.time() - self._last_failure_time >= self._recovery_timeout:
                    self._state = CircuitState.HALF_OPEN
                    logger.info(
                        "Circuit breaker HALF_OPEN — allowing probe call "
                        "(after %.0fs recovery timeout)",
                        self._recovery_timeout,
                    )
            return self._state

    @property
    def is_open(self) -> bool:
        """True when calls should be skipped (circuit is open)."""
        return self.state == CircuitState.OPEN

    def record_success(self) -> None:
        """Call succeeded — close the circuit."""
        with self._lock:
            if self._state != CircuitState.CLOSED:
                logger.info("Circuit breaker CLOSED — embedding provider recovered")
            self._state = CircuitState.CLOSED
            self._consecutive_failures = 0

    def record_failure(self) -> None:
        """Call failed — potentially open the circuit."""
        with self._lock:
            self._consecutive_failures += 1
            self._last_failure_time = time.time()
            if self._consecutive_failures >= self._failure_threshold:
                if self._state != CircuitState.OPEN:
                    logger.warning(
                        "Circuit breaker OPEN — %d consecutive failures, "
                        "falling back to text-only for %.0fs",
                        self._consecutive_failures, self._recovery_timeout,
                    )
                self._state = CircuitState.OPEN

    def reset(self) -> None:
        """Manually reset the circuit to closed."""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._consecutive_failures = 0
            self._last_failure_time = 0.0
