from collections.abc import AsyncIterator

import pytest

from qs.core.agents.deep_agent.stream_event import DeepAgentEvent
from qs.core.models.client_request import ClientRequest
from qs.core.models.provider_request import ProviderRequest
from qs.core.models.provider_stream_event import (
    ProviderStreamEvent,
    ProviderStreamEventType,
)
from qs.core.models.system_message import SystemMessage
from qs.core.models.user_message import UserMessage
from qs.interfaces.event_handler import DeepAgentEventHandler, ProviderEventHandler


class CollectProviderEvents(ProviderEventHandler):
    def __init__(self) -> None:
        self.events: list[ProviderStreamEvent] = []

    def handle(self, event: ProviderStreamEvent) -> None:
        self.events.append(event)


def test_provider_event_handler_is_validated_and_called() -> None:
    handler = CollectProviderEvents()
    request = ProviderRequest([SystemMessage("rules")], on_event=handler)
    client_request = ClientRequest(UserMessage("question"), on_event=handler)
    event = ProviderStreamEvent(ProviderStreamEventType.START)

    request.execute_on_stream_event(event)

    assert client_request.on_event is handler
    assert handler.events == [event]


def test_event_handler_rejects_wrong_event_annotation() -> None:
    class Invalid(ProviderEventHandler):
        def handle(self, event: str) -> None:
            pass

    with pytest.raises(TypeError, match="must define handle"):
        ClientRequest(UserMessage("question"), on_event=Invalid())


def test_event_handler_rejects_raw_function() -> None:
    with pytest.raises(TypeError, match="ProviderEventHandler instance"):
        ClientRequest(UserMessage("question"), on_event=lambda event: None)


def test_event_handler_rejects_async_method() -> None:
    class AsyncHandler(ProviderEventHandler):
        async def handle(self, event: ProviderStreamEvent) -> None:
            pass

    with pytest.raises(TypeError, match="synchronous method"):
        ProviderRequest([SystemMessage("rules")], on_event=AsyncHandler())


def test_event_handler_rejects_async_generator_method() -> None:
    class AsyncGeneratorHandler(ProviderEventHandler):
        async def handle(
            self, event: ProviderStreamEvent
        ) -> AsyncIterator[ProviderStreamEvent]:
            yield event

    with pytest.raises(TypeError, match="synchronous method"):
        ProviderRequest([SystemMessage("rules")], on_event=AsyncGeneratorHandler())


def test_event_handler_rejects_non_callable_handle() -> None:
    class ShadowedHandler(ProviderEventHandler):
        def handle(self, event: ProviderStreamEvent) -> None:
            pass

    shadowed = ShadowedHandler()
    shadowed.handle = None
    with pytest.raises(TypeError, match="handle must be callable"):
        ProviderRequest([SystemMessage("rules")], on_event=shadowed)


def test_event_handler_requires_one_positional_event() -> None:
    class MissingEvent(ProviderEventHandler):
        def handle(self) -> None:
            pass

    with pytest.raises(TypeError, match="exactly one event"):
        ProviderRequest([SystemMessage("rules")], on_event=MissingEvent())


def test_event_handler_rejects_unresolved_annotation() -> None:
    class UnresolvedEvent(ProviderEventHandler):
        def handle(self, event: "MissingEventType") -> None:  # noqa: F821
            pass

    with pytest.raises(TypeError, match="annotations could not be resolved"):
        ProviderRequest([SystemMessage("rules")], on_event=UnresolvedEvent())


def test_deep_agent_handler_contract_is_validated() -> None:
    class Handler(DeepAgentEventHandler):
        def handle(self, event: DeepAgentEvent) -> None:
            pass

    DeepAgentEventHandler.validate_instance(Handler())

    with pytest.raises(TypeError, match="DeepAgentEventHandler instance"):
        DeepAgentEventHandler.validate_instance(object())
