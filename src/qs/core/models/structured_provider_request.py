from __future__ import annotations

from pydantic import BaseModel

from qs.core.models.provider_request import ProviderRequest
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
    ) -> None:
        super().__init__(
            messages,
            extra=extra,
            tool_choice=tool_choice,
            tools=tools,
        )
        self._response_model = response_model
        self._strict = strict
        self._check_inputs()

    def _check_inputs(self) -> None:
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
