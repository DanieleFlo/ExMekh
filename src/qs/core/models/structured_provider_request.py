from __future__ import annotations

from collections.abc import Callable

from pydantic import BaseModel

from qs.core.models.provider_request import ProviderRequest
from qs.core.models.provider_stream_event import ProviderStreamEvent
from qs.core.models.tools import Tools
from qs.interfaces.message import Message


class StructuredProviderRequest[ResponseModel: BaseModel](ProviderRequest):
    def __init__(
        self,
        messages: list[Message],
        response_model: type[ResponseModel],
        *,
        strict: bool = True,
        extra: dict | None = None,
        tool_choice: str | None = None,
        tools: Tools | None = None,
        stream: bool = False,
        on_event: Callable[[ProviderStreamEvent], None] | None = None,
    ) -> None:
        super().__init__(
            messages,
            extra=extra,
            tool_choice=tool_choice,
            tools=tools,
            stream=stream,
            on_event=on_event,
        )
        self._response_model = response_model
        self._strict = strict
        self._check_structured_inputs()

    def _check_structured_inputs(self) -> None:
        if not isinstance(self._response_model, type) or not issubclass(
            self._response_model, BaseModel
        ):
            raise TypeError("Response model must be a Pydantic BaseModel type")
        if not isinstance(self._strict, bool):
            raise TypeError("Strict must be a boolean")

    def get_response_model(self) -> type[ResponseModel]:
        return self._response_model

    def get_strict(self) -> bool:
        return self._strict
