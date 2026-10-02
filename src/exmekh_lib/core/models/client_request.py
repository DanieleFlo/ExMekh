from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from exmekh_lib.interfaces.event_handler import ProviderEventHandler
from exmekh_lib.interfaces.message import Message


@dataclass(frozen=True)
class ClientRequest:
    message: Message[object]
    extra: Mapping[str, object] | None = None
    tool_choice: str | None = None
    streaming: bool = False
    on_event: ProviderEventHandler | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.message, Message):
            raise TypeError("Expected a Message instance")
        if self.extra is not None and not isinstance(self.extra, Mapping):
            raise TypeError("Extra provider data must be a mapping")
        if self.tool_choice is not None and not isinstance(self.tool_choice, str):
            raise TypeError("Tool choice must be a string")
        if not isinstance(self.streaming, bool):
            raise TypeError("Streaming must be a boolean")
        if self.on_event is not None:
            ProviderEventHandler.validate_instance(self.on_event)

        if self.extra is not None:
            object.__setattr__(self, "extra", dict(self.extra))
