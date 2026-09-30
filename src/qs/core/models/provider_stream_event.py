from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from qs.core.models.model_usage import ModelUsage
from qs.core.models.user_message import UserMessage


class ProviderStreamEventType(StrEnum):
    START = "start"
    REASONING = "reasoning"
    RESPONSE = "response"
    TOTALS = "totals"
    END = "end"
    STEERING_ADDED = "steering_added"


@dataclass(frozen=True, slots=True)
class ProviderStreamEvent:
    type: ProviderStreamEventType
    content: str | None = None
    usage: ModelUsage | None = None
    message: UserMessage | None = None

    def __post_init__(self) -> None:
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
