from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from enum import StrEnum

from exmekh_lib.config.model_text import MODEL_TEXT
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
                    MODEL_TEXT.interactive_events_require_structured_control_metadata
                )
            object.__setattr__(self, "interactive", deepcopy(self.interactive))
        elif self.interactive is not None:
            raise ValueError(
                MODEL_TEXT.only_interactive_events_can_contain_control_metadata
            )
        if not isinstance(self.type, ProviderStreamEventType):
            raise TypeError(
                MODEL_TEXT.stream_event_type_must_be_a_providerstreameventtype
            )
        if self.content is not None and not isinstance(self.content, str):
            raise TypeError(MODEL_TEXT.stream_event_content_must_be_a_string_or_none)
        if self.usage is not None and not isinstance(self.usage, ModelUsage):
            raise TypeError(MODEL_TEXT.stream_event_usage_must_be_a_modelusage_or_none)

        content_types = {
            ProviderStreamEventType.REASONING,
            ProviderStreamEventType.RESPONSE,
        }
        if self.type in content_types and self.content is None:
            raise ValueError(MODEL_TEXT.reasoning_and_response_events_require_content)
        if self.type not in content_types and self.content is not None:
            raise ValueError(
                MODEL_TEXT.only_reasoning_and_response_events_can_contain_content
            )
        if self.type is ProviderStreamEventType.TOTALS and self.usage is None:
            raise ValueError(MODEL_TEXT.totals_events_require_usage)
        if self.type is not ProviderStreamEventType.TOTALS and self.usage is not None:
            raise ValueError(MODEL_TEXT.only_totals_events_can_contain_usage)
        if self.type is ProviderStreamEventType.STEERING_ADDED:
            if not isinstance(self.message, UserMessage):
                raise TypeError(MODEL_TEXT.steering_events_require_a_usermessage)
            object.__setattr__(self, "message", self.message.copy())
        elif self.message is not None:
            raise ValueError(MODEL_TEXT.only_steering_events_can_contain_a_message)
