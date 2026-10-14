from __future__ import annotations

from typing import Any

from exmekh_lib.config.provider_text import PROVIDER_TEXT
from exmekh_lib.core.models.provider_request import ProviderRequest
from exmekh_lib.core.models.reasoning_effort import ReasoningEffort
from exmekh_lib.core.providers.openai_response_tool_mapper import (
    OpenAIResponseToolMapper,
)
from exmekh_lib.core.providers.to_openai_response_mapper import (
    ToOpenAIResponseMapper,
)
from exmekh_lib.interfaces.message import Message


class ToOpenAIResponseRequestMapper:
    @classmethod
    def map_request(
        cls,
        request: ProviderRequest,
        *,
        model: str,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> dict[str, Any]:
        if not isinstance(request, ProviderRequest):
            raise TypeError(PROVIDER_TEXT.expected_a_providerrequest_instance)

        payload: dict[str, Any] = {
            "model": model,
            "input": cls._map_messages(request.get_messages()),
            "stream": request.get_streaming(),
        }

        reasoning_effort = request.get_reasoning_effort_override() or reasoning_effort
        if reasoning_effort is not None:
            effort = cls._get_reasoning_effort_value(reasoning_effort)
            payload["reasoning"] = {"effort": effort}
            if effort != "none":
                payload["reasoning"]["summary"] = "auto"

        extra = request.get_extra()
        if extra is not None:
            payload["extra_body"] = extra

        tool_choice = request.get_tool_choice()
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

        tools = request.get_tools()
        if tools is not None:
            mapped_tools = OpenAIResponseToolMapper.map_tools(tools)
            if mapped_tools:
                payload["tools"] = mapped_tools

        return payload

    @staticmethod
    def _get_reasoning_effort_value(reasoning_effort: ReasoningEffort) -> str:
        if isinstance(reasoning_effort, str):
            return reasoning_effort
        if isinstance(reasoning_effort, ReasoningEffort):
            return reasoning_effort.value
        raise TypeError(PROVIDER_TEXT.reasoning_effort_must_be_a_string_or_an_instance)

    @classmethod
    def _map_messages(cls, messages: list[Message[object]]) -> list[dict[str, Any]]:
        mapped: list[dict[str, Any]] = []
        for message in messages:
            mapped.extend(ToOpenAIResponseMapper.map_message(message))
        return mapped
