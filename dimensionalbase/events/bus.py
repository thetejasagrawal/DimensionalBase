"""
EventBus — pub/sub system for DimensionalBase.

Agents subscribe to glob patterns on knowledge paths.
The system fires events on writes, conflicts, gaps, and staleness.
Simple, synchronous, no external dependencies.
"""

from __future__ import annotations

import fnmatch
import logging
import time
import uuid
from typing import Callable, Dict, List, Optional

from dimensionalbase.core.types import Event, EventType, Subscription

logger = logging.getLogger("dimensionalbase.events")


class EventBus:
    """Publish-subscribe event system with glob pattern matching.

    Usage:
        bus = EventBus()
        sub = bus.subscribe("task/**", "planner", callback)
        bus.emit(Event(type=EventType.CHANGE, path="task/auth/status"))
        bus.unsubscribe(sub)
    """

    def __init__(self, max_history: int = 1000):
        self._subscriptions: Dict[str, Subscription] = {}
        self._event_history: List[Event] = []
        self._max_history = max_history
        self._dropped_count = 0

    def subscribe(
        self,
        pattern: str,
        subscriber: str,
        callback: Callable[[Event], None],
    ) -> Subscription:
        """Subscribe to events matching a glob pattern.

        Args:
            pattern:    Glob pattern (e.g., 'task/**', 'task/auth/*').
            subscriber: Name of the subscribing agent.
            callback:   Function called when a matching event fires.

        Returns:
            Subscription handle (pass to unsubscribe to stop).
        """
        sub_id = uuid.uuid4().hex[:12]
        sub = Subscription(
            id=sub_id,
            pattern=pattern,
            subscriber=subscriber,
            callback=callback,
            active=True,
        )
        self._subscriptions[sub_id] = sub
        logger.debug(f"Subscription created: {subscriber} -> {pattern} (id={sub_id})")
        return sub

    def unsubscribe(self, subscription: Subscription) -> bool:
        """Cancel a subscription.

        Args:
            subscription: The subscription handle returned by subscribe().

        Returns:
            True if the subscription was found and removed.
        """
        if subscription.id in self._subscriptions:
            self._subscriptions[subscription.id].active = False
            del self._subscriptions[subscription.id]
            logger.debug(f"Subscription removed: {subscription.id}")
            return True
        return False

    def emit(self, event: Event) -> int:
        """Fire an event to all matching subscribers.

        Args:
            event: The event to emit.

        Returns:
            Number of subscribers notified.
        """
        if event.timestamp == 0.0:
            event.timestamp = time.time()

        # Store in history
        self._event_history.append(event)
        if len(self._event_history) > self._max_history:
            dropped = len(self._event_history) - self._max_history
            self._dropped_count += dropped
            logger.warning(
                "Event history full (%d max). Dropping %d oldest event(s). "
                "Total dropped: %d",
                self._max_history, dropped, self._dropped_count,
            )
            self._event_history = self._event_history[-self._max_history:]

        notified = 0
        for sub in list(self._subscriptions.values()):
            if not sub.active:
                continue
            if self._matches(sub.pattern, event.path):
                try:
                    sub.callback(event)
                    notified += 1
                except Exception as e:
                    logger.error(
                        f"Subscriber {sub.subscriber} callback failed for "
                        f"{event.type.value} at {event.path}: {e}"
                    )
        return notified

    def get_history(
        self,
        pattern: Optional[str] = None,
        event_type: Optional[EventType] = None,
        limit: int = 50,
    ) -> List[Event]:
        """Get recent events, optionally filtered.

        Args:
            pattern:    Only events matching this glob pattern.
            event_type: Only events of this type.
            limit:      Max events to return.

        Returns:
            List of events, newest first.
        """
        results = []
        for event in reversed(self._event_history):
            if pattern and not self._matches(pattern, event.path):
                continue
            if event_type and event.type != event_type:
                continue
            results.append(event)
            if len(results) >= limit:
                break
        return results

    @property
    def subscription_count(self) -> int:
        """Number of active subscriptions."""
        return len(self._subscriptions)

    @property
    def dropped_event_count(self) -> int:
        """Total events dropped due to history buffer overflow."""
        return self._dropped_count

    @staticmethod
    def _matches(pattern: str, path: str) -> bool:
        """Check if a path matches a glob pattern.  Delegates to canonical DBPS matcher."""
        from dimensionalbase.core.matching import dbps_match
        return dbps_match(pattern, path)
