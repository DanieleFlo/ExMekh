from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum

from exmekh_lib.core.models.model_usage import ModelUsage
from exmekh_lib.core.models.user_message import UserMessage


class ProviderStreamEventType(StrEnum):
    START = "start"
    REASONING = "reasoning"
    RESPONSE = "response"
    TOTALS = "totals"
    END = "end"
    STEERING_ADDED = "steering_added"
    INTERACTIVE = "interactive"


@dataclass(frozen=True, slots=True)
class ProviderStreamEvent:
    type: ProviderStreamEventType
    content: str | None = None
    usage: ModelUsage | None = None
    message: UserMessage | None = None
    interactive: dict[str, object] | None = None

    def __post_init__(self) -> None:
        if self.type is ProviderStreamEventType.INTERACTIVE:
            if not isinstance(self.interactive, dict):
                raise TypeError(
                    "Interactive events require structured control metadata"
                )
            object.__setattr__(self, "interactive", deepcopy(self.interactive))
        elif self.interactive is not None:
            raise ValueError("Only interactive events can contain control metadata")
        if not isinstance(self.type, ProviderStreamEventType):
            raise TypeError("Stream event type must be a ProviderStreamEventType")
        if self.content is not None and not isinstance(self.content, str):
            raise TypeError("Stream event content must be a string or None")
        if self.usage is not None and not isinstance(self.usage, ModelUsage):
            raise TypeError("Stream event usage must be a ModelUsage or None")

        content_types = {
            ProviderStreamEventType.REASONING,
            ProviderStreamEventType.RESPONSE,
        }
        if self.type in content_types and self.content is None:
            raise ValueError("Reasoning and response events require content")
        if self.type not in content_types and self.content is not None:
            raise ValueError("Only reasoning and response events can contain content")
        if self.type is ProviderStreamEventType.TOTALS and self.usage is None:
            raise ValueError("Totals events require usage")
        if self.type is not ProviderStreamEventType.TOTALS and self.usage is not None:
            raise ValueError("Only totals events can contain usage")
        if self.type is ProviderStreamEventType.STEERING_ADDED:
            if not isinstance(self.message, UserMessage):
                raise TypeError("Steering events require a UserMessage")
            object.__setattr__(self, "message", self.message.copy())
        elif self.message is not None:
            raise ValueError("Only steering events can contain a message")
