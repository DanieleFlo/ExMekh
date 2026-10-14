from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel

from exmekh_lib.config.model_text import MODEL_TEXT
from exmekh_lib.core.models.provider_request import ProviderRequest
from exmekh_lib.core.models.tools import Tools
from exmekh_lib.interfaces.event_handler import ProviderEventHandler
from exmekh_lib.interfaces.message import Message


class StructuredProviderRequest[ResponseModel: BaseModel](ProviderRequest):
    def __init__(
        self,
        messages: list[Message[object]],
        response_model: type[ResponseModel],
        *,
        strict: bool = True,
        extra: Mapping[str, object] | None = None,
        tool_choice: str | None = None,
        tools: Tools | None = None,
        stream: bool = False,
        on_event: ProviderEventHandler | None = None,
        reasoning_effort_override: str | None = None,
    ) -> None:
        super().__init__(
            messages,
            extra=extra,
            tool_choice=tool_choice,
            tools=tools,
            stream=stream,
            on_event=on_event,
            reasoning_effort_override=reasoning_effort_override,
        )
        self._response_model = response_model
        self._strict = strict
        self._check_structured_inputs()

    def _check_structured_inputs(self) -> None:
        if not isinstance(self._response_model, type) or not issubclass(
            self._response_model, BaseModel
        ):
            raise TypeError(MODEL_TEXT.response_model_must_be_a_pydantic_basemodel_type)
        if not isinstance(self._strict, bool):
            raise TypeError(MODEL_TEXT.strict_must_be_a_boolean)

    def get_response_model(self) -> type[ResponseModel]:
        return self._response_model

    def get_strict(self) -> bool:
        return self._strict
