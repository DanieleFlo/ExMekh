from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from exmekh_lib.config.model_text import MODEL_TEXT
from exmekh_lib.interfaces.event_handler import ProviderEventHandler
from exmekh_lib.interfaces.message import Message


@dataclass(frozen=True)
class ClientRequest:
    message: Message[object]
    extra: Mapping[str, object] | None = None
    tool_choice: str | None = None
    streaming: bool = False
    on_event: ProviderEventHandler | None = None
    reasoning_effort_override: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.message, Message):
            raise TypeError(MODEL_TEXT.expected_a_message_instance)
        if self.extra is not None and not isinstance(self.extra, Mapping):
            raise TypeError(MODEL_TEXT.extra_provider_data_must_be_a_mapping)
        if self.tool_choice is not None and not isinstance(self.tool_choice, str):
            raise TypeError(MODEL_TEXT.tool_choice_must_be_a_string)
        if not isinstance(self.streaming, bool):
            raise TypeError(MODEL_TEXT.streaming_must_be_a_boolean)
        if self.on_event is not None:
            ProviderEventHandler.validate_instance(self.on_event)
        if self.reasoning_effort_override is not None and (
            not isinstance(self.reasoning_effort_override, str)
            or not self.reasoning_effort_override.strip()
        ):
            raise ValueError(
                MODEL_TEXT.reasoning_effort_override_must_be_a_non_empty_string
            )

        if self.extra is not None:
            object.__setattr__(self, "extra", dict(self.extra))
