from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from qs.core.models.provider_stream_event import ProviderStreamEvent
from qs.interfaces.message import Message


@dataclass(frozen=True)
class ClientRequest:
    message: Message
    extra: dict | None = None
    tool_choice: str | None = None
    streaming: bool = False
    on_event: Callable[[ProviderStreamEvent], None] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.message, Message):
            raise TypeError("Expected a Message instance")
        if self.extra is not None and not isinstance(self.extra, dict):
            raise TypeError("Extra provider data must be a dictionary")
        if self.tool_choice is not None and not isinstance(self.tool_choice, str):
            raise TypeError("Tool choice must be a string")
        if not isinstance(self.streaming, bool):
            raise TypeError("Streaming must be a boolean")
        if self.on_event is not None and not callable(self.on_event):
            raise TypeError("on_event must be a callable if provided")

        if self.extra is not None:
            object.__setattr__(self, "extra", self.extra.copy())
