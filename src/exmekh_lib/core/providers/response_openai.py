from __future__ import annotations

from collections.abc import Iterable
from typing import Any, cast

from openai import OpenAI
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseFailedEvent,
    ResponseIncompleteEvent,
    ResponseStreamEvent,
)
from pydantic import BaseModel

from exmekh_lib.core.models.provider_request import ProviderRequest
from exmekh_lib.core.models.provider_response import ProviderResponse
from exmekh_lib.core.models.provider_stream_event import (
    ProviderStreamEvent,
    ProviderStreamEventType,
)
from exmekh_lib.core.models.reasoning_effort import ReasoningEffort
from exmekh_lib.core.models.structured_assistant_message import (
    StructuredAssistantMessage,
)
from exmekh_lib.core.models.structured_provider_request import StructuredProviderRequest
from exmekh_lib.core.models.structured_provider_response import (
    StructuredProviderResponse,
)
from exmekh_lib.core.models.tools import Tools
from exmekh_lib.core.providers.from_openai_response_mapper import (
    FromOpenAIResponseMapper,
)
from exmekh_lib.core.providers.from_openai_response_stream_mapper import (
    FromOpenAIResponseStreamMapper,
)
from exmekh_lib.core.providers.openai_response_format_mapper import (
    OpenAIResponseFormatMapper,
)
from exmekh_lib.core.providers.openai_response_tool_mapper import (
    OpenAIResponseToolMapper,
)
from exmekh_lib.core.providers.to_openai_response_request_mapper import (
    ToOpenAIResponseRequestMapper,
)
from exmekh_lib.utils.json import serialize_json


class ResponseOpenAI:
    def __init__(
        self,
        client: OpenAI,
        model: str,
        reasoning_effort: ReasoningEffort | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._reasoning_effort = reasoning_effort
        self._check_inputs()

    def _check_inputs(self) -> None:
        if not isinstance(self._client, OpenAI):
            raise TypeError("Client must be an instance of OpenAI")
        if not isinstance(self._model, str):
            raise TypeError("Model must be a string")
        if self._reasoning_effort is not None and not isinstance(
            self._reasoning_effort,
            ReasoningEffort,
        ):
            raise TypeError("Reasoning effort must be a ReasoningEffort instance")

    def call(self, request: ProviderRequest) -> ProviderResponse:
        response = self._execute_request(request)
        return self._parse_response(response, request)

    def get_tools_text(self, tools: Tools | None) -> str:
        mapped_tools = (
            OpenAIResponseToolMapper.map_tools(tools) if tools is not None else []
        )
        return serialize_json(mapped_tools) if mapped_tools else ""

    def call_structured[ResponseModel: BaseModel](
        self,
        request: StructuredProviderRequest[ResponseModel],
    ) -> StructuredProviderResponse[ResponseModel]:
        response = self._execute_structured_request(request)
        return self._parse_structured_response(response, request)

    def _execute_request(self, request: ProviderRequest) -> Response:
        payload = ToOpenAIResponseRequestMapper.map_request(
            request,
            model=self._model,
            reasoning_effort=self._reasoning_effort,
        )
        return self._execute_payload(request, payload)

    def _execute_structured_request[ResponseModel: BaseModel](
        self,
        request: StructuredProviderRequest[ResponseModel],
    ) -> Response:
        payload = ToOpenAIResponseRequestMapper.map_request(
            request,
            model=self._model,
            reasoning_effort=self._reasoning_effort,
        )
        payload["text"] = OpenAIResponseFormatMapper.map(
            request.get_response_model(),
            strict=request.get_strict(),
        )
        return self._execute_payload(request, payload)

    def _execute_payload(
        self,
        request: ProviderRequest,
        payload: dict[str, Any],
    ) -> Response:
        response = self._client.responses.create(**payload)
        if payload.get("stream") is not True:
            return cast(Response, response)
        return self._consume_stream(
            cast(Iterable[ResponseStreamEvent], response),
            request,
        )

    @classmethod
    def _consume_stream(
        cls,
        stream: Iterable[ResponseStreamEvent],
        request: ProviderRequest,
    ) -> Response:
        response: Response | None = None
        request.execute_on_stream_event(
            ProviderStreamEvent(ProviderStreamEventType.START)
        )

        try:
            for event in stream:
                for mapped_event in FromOpenAIResponseStreamMapper.map_event(event):
                    request.execute_on_stream_event(mapped_event)

                if isinstance(
                    event,
                    (
                        ResponseCompletedEvent,
                        ResponseFailedEvent,
                        ResponseIncompleteEvent,
                    ),
                ):
                    response = event.response
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                close()

        if response is None:
            raise RuntimeError("No completed response returned from the provider")

        cls._ensure_completed(response)
        request.execute_on_stream_event(
            ProviderStreamEvent(ProviderStreamEventType.END)
        )
        return response

    @staticmethod
    def _ensure_completed(response: Response) -> None:
        if response.error is not None:
            raise RuntimeError(f"Provider response failed: {response.error.message}")
        if response.status != "completed":
            raise RuntimeError(f"Provider response did not complete: {response.status}")

    def _parse_response(
        self,
        response: Response,
        request: ProviderRequest,
    ) -> ProviderResponse:
        self._ensure_completed(response)
        response_content = FromOpenAIResponseMapper.map_assistant_message(
            response,
            tools=request.get_tools(),
        )
        usage = FromOpenAIResponseMapper.extract_usage(response)
        thinking = FromOpenAIResponseMapper.extract_thinking(response)
        return ProviderResponse(
            response=response_content,
            thinking=thinking,
            usage=usage,
        )

    def _parse_structured_response[ResponseModel: BaseModel](
        self,
        response: Response,
        request: StructuredProviderRequest[ResponseModel],
    ) -> StructuredProviderResponse[ResponseModel]:
        self._ensure_completed(response)
        assistant_message = FromOpenAIResponseMapper.map_assistant_message(
            response,
            tools=request.get_tools(),
        )
        response_model = request.get_response_model()
        content = response_model.model_validate_json(
            assistant_message.get_message() or "{}",
            strict=request.get_strict(),
        )
        return StructuredProviderResponse(
            response=StructuredAssistantMessage(
                content,
                tool_calls=assistant_message.get_tool_calls(),
            ),
            thinking=FromOpenAIResponseMapper.extract_thinking(response),
            usage=FromOpenAIResponseMapper.extract_usage(response),
        )
