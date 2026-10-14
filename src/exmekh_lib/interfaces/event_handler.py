from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, get_type_hints

from exmekh_lib.config.interface_text import INTERFACE_TEXT
from exmekh_lib.core.models.provider_stream_event import ProviderStreamEvent

if TYPE_CHECKING:
    from exmekh_lib.core.agents.deep_agent.stream_event import DeepAgentEvent


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
        from exmekh_lib.core.agents.deep_agent.stream_event import DeepAgentEvent  # noqa: I001, PLC0415

        _validate_handle(handler, cls, DeepAgentEvent, "DeepAgentEvent")


def _validate_handle(
    handler: object,
    interface: type[ProviderEventHandler] | type[DeepAgentEventHandler],
    event_type: object,
    event_name: str,
) -> None:
    if not isinstance(handler, interface):
        raise TypeError(
            INTERFACE_TEXT.event_handler_must_be_a_instance.format(
                interface_name=interface.__name__
            )
        )
    method = handler.handle
    if not callable(method):
        raise TypeError(INTERFACE_TEXT.event_handler_handle_must_be_callable)
    if (
        inspect.iscoroutinefunction(method)
        or inspect.isgeneratorfunction(method)
        or inspect.isasyncgenfunction(method)
    ):
        raise TypeError(
            INTERFACE_TEXT.event_handler_handle_must_be_a_synchronous_method
        )
    signature = inspect.signature(method)
    parameters = tuple(signature.parameters.values())
    if len(parameters) != 1 or parameters[0].kind not in (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    ):
        raise TypeError(INTERFACE_TEXT.event_handler_must_accept_exactly_one_event)
    try:
        hints = get_type_hints(method)
    except (NameError, TypeError) as error:
        raise TypeError(
            INTERFACE_TEXT.event_handler_annotations_could_not_be_resolved
        ) from error
    if hints.get(parameters[0].name) != event_type or hints.get("return") is not type(
        None
    ):
        raise TypeError(
            INTERFACE_TEXT.event_handler_must_define_handle_event_none.format(
                event_name=event_name
            )
        )
