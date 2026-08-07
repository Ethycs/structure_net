"""
Event bus for the StructureNet kernel.

Provides decoupled publish/subscribe messaging between kernel services and
components. Handlers are isolated: one failing subscriber never prevents the
event from reaching the others.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional


EventHandler = Callable[["Event"], None]

WILDCARD = "*"


@dataclass
class Event:
    """A single kernel event."""

    name: str
    payload: Dict[str, Any] = field(default_factory=dict)
    source: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


class EventBus:
    """Synchronous publish/subscribe event bus.

    Subscribers register a handler for an exact event name or for the
    wildcard ``"*"``. Publishing calls every matching handler in registration
    order; handler exceptions are logged (when a logger is available) and
    swallowed so remaining subscribers still run.
    """

    def __init__(self, logger: Any = None, history_limit: int = 256):
        self._logger = logger
        self._subscribers: Dict[str, List[EventHandler]] = {}
        self._history: List[Event] = []
        self._history_limit = int(history_limit)

    def subscribe(self, event_name: str, handler: EventHandler) -> None:
        """Register a handler for ``event_name`` (or ``"*"`` for all events)."""
        self._subscribers.setdefault(event_name, []).append(handler)

    def unsubscribe(self, event_name: str, handler: EventHandler) -> bool:
        """Remove a previously registered handler. Returns True if removed."""
        handlers = self._subscribers.get(event_name, [])
        try:
            handlers.remove(handler)
            return True
        except ValueError:
            return False

    def publish(
        self,
        event_name: str,
        payload: Optional[Dict[str, Any]] = None,
        source: Optional[str] = None,
    ) -> Event:
        """Publish an event to exact-name and wildcard subscribers."""
        event = Event(name=event_name, payload=dict(payload or {}), source=source)
        self._history.append(event)
        if len(self._history) > self._history_limit:
            self._history = self._history[-self._history_limit :]
        for handler in list(self._subscribers.get(event_name, ())) + list(
            self._subscribers.get(WILDCARD, ())
        ):
            try:
                handler(event)
            except Exception as error:  # noqa: BLE001 - isolation is the contract
                if self._logger is not None:
                    try:
                        self._logger.error(
                            f"Event handler failed for '{event_name}': {error}"
                        )
                    except Exception:  # noqa: BLE001 - logging must never raise
                        pass
        return event

    @property
    def history(self) -> List[Event]:
        """Recent events, oldest first (bounded by ``history_limit``)."""
        return list(self._history)

    def subscriber_count(self, event_name: str) -> int:
        return len(self._subscribers.get(event_name, ()))
