from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, get_type_hints

from qs.core.models.provider_stream_event import ProviderStreamEvent

if TYPE_CHECKING:
    from qs.core.agents.deep_agent.stream_event import DeepAgentEvent


class ProviderEventHandler(ABC):
    """Receive one provider streaming event."""

    @abstractmethod
    def handle(self, event: ProviderStreamEvent) -> None:
        """Handle a provider event."""

    @classmethod
    def validate_instance(cls, handler: object) -> None:
        _validate_handle(handler, cls, ProviderStreamEvent, "ProviderStreamEvent")


class DeepAgentEventHandler(ABC):
    """Receive one agent streaming or execution event."""

    @abstractmethod
    def handle(self, event: DeepAgentEvent) -> None:
        """Handle an agent event."""

    @classmethod
    def validate_instance(cls, handler: object) -> None:
        # The agent package exports this interface, so a module-level import cycles.
        from qs.core.agents.deep_agent.stream_event import DeepAgentEvent  # noqa: I001, PLC0415

        _validate_handle(handler, cls, DeepAgentEvent, "DeepAgentEvent")


def _validate_handle(
    handler: object,
    interface: type[ProviderEventHandler] | type[DeepAgentEventHandler],
    event_type: object,
    event_name: str,
) -> None:
    if not isinstance(handler, interface):
        raise TypeError(f"event handler must be a {interface.__name__} instance")
    method = handler.handle
    if not callable(method):
        raise TypeError("Event handler handle must be callable")
    if (
        inspect.iscoroutinefunction(method)
        or inspect.isgeneratorfunction(method)
        or inspect.isasyncgenfunction(method)
    ):
        raise TypeError("Event handler handle must be a synchronous method")
    signature = inspect.signature(method)
    parameters = tuple(signature.parameters.values())
    if len(parameters) != 1 or parameters[0].kind not in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    ):
        raise TypeError("Event handler must accept exactly one event")
    try:
        hints = get_type_hints(method)
    except (NameError, TypeError) as error:
        raise TypeError("Event handler annotations could not be resolved") from error
    if hints.get(parameters[0].name) != event_type or hints.get("return") is not type(
        None
    ):
        raise TypeError(
            f"Event handler must define handle(event: {event_name}) -> None"
        )
