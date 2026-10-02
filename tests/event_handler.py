from __future__ import annotations

from collections.abc import Callable

from exmekh_lib.core.agents.deep_agent.stream_event import DeepAgentEvent
from exmekh_lib.core.models.provider_stream_event import ProviderStreamEvent
from exmekh_lib.interfaces.event_handler import (
    DeepAgentEventHandler,
    ProviderEventHandler,
)


class CallbackProviderEventHandler(ProviderEventHandler):
    def __init__(self, callback: Callable[[ProviderStreamEvent], None]) -> None:
        self._callback = callback

    def handle(self, event: ProviderStreamEvent) -> None:
        self._callback(event)


class CallbackAgentEventHandler(DeepAgentEventHandler):
    def __init__(self, callback: Callable[[DeepAgentEvent], None]) -> None:
        self._callback = callback

    def handle(self, event: DeepAgentEvent) -> None:
        self._callback(event)


def as_provider_handler(
    callback: Callable[[ProviderStreamEvent], None] | None,
) -> ProviderEventHandler | None:
    if callback is None:
        return None
    return CallbackProviderEventHandler(callback)


def as_agent_handler(
    callback: Callable[[DeepAgentEvent], None] | None,
) -> DeepAgentEventHandler | None:
    if callback is None:
        return None
    return CallbackAgentEventHandler(callback)
