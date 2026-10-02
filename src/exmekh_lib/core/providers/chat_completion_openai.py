from __future__ import annotations

from collections.abc import Iterable
from typing import Any, cast

from openai import OpenAI
from openai.lib.streaming.chat import ChatCompletionStreamState
from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion import ChatCompletion
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
from exmekh_lib.core.providers.from_openai_chat_completion_mapper import (
    FromOpenAIChatCompletionMapper,
)
from exmekh_lib.core.providers.from_openai_chat_completion_stream_mapper import (
    FromOpenAIChatCompletionStreamMapper,
)
from exmekh_lib.core.providers.openai_chat_completion_response_format_mapper import (
    OpenAIChatCompletionResponseFormatMapper,
)
from exmekh_lib.core.providers.openai_chat_completion_tool_mapper import (
    OpenAIChatCompletionToolMapper,
)
from exmekh_lib.core.providers.to_openai_chat_completion_request_mapper import (
    ToOpenAIChatCompletionRequestMapper,
)
from exmekh_lib.utils.json import serialize_json


class ChatCompletionOpenAI:
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
            self._reasoning_effort, ReasoningEffort
        ):
            raise TypeError("Reasoning effort must be a ReasoningEffort instance")

    def call(self, request: ProviderRequest) -> ProviderResponse:
        response = self._execute_request(request)
        return self._parse_response(response, request)

    def get_tools_text(self, tools: Tools | None) -> str:
        mapped_tools = (
            OpenAIChatCompletionToolMapper.map_tools(tools) if tools is not None else []
        )
        return serialize_json(mapped_tools) if mapped_tools else ""

    def call_structured[ResponseModel: BaseModel](
        self,
        request: StructuredProviderRequest[ResponseModel],
    ) -> StructuredProviderResponse[ResponseModel]:
        response = self._execute_structured_request(request)
        return self._parse_structured_response(response, request)

    def _execute_request(self, request: ProviderRequest) -> ChatCompletion:
        payload = ToOpenAIChatCompletionRequestMapper.map_request(
            request,
            model=self._model,
            reasoning_effort=self._reasoning_effort,
        )
        return self._execute_payload(request, payload)

    def _execute_structured_request[ResponseModel: BaseModel](
        self,
        request: StructuredProviderRequest[ResponseModel],
    ) -> ChatCompletion:
        payload = ToOpenAIChatCompletionRequestMapper.map_request(
            request,
            model=self._model,
            reasoning_effort=self._reasoning_effort,
        )

        payload["response_format"] = OpenAIChatCompletionResponseFormatMapper.map(
            request.get_response_model(),
            strict=request.get_strict(),
        )
        return self._execute_payload(request, payload)

    def _execute_payload(
        self,
        request: ProviderRequest,
        payload: dict[str, Any],
    ) -> ChatCompletion:
        response = self._client.chat.completions.create(**payload)
        if payload.get("stream") is not True:
            return cast(ChatCompletion, response)

        return self._consume_stream(
            cast(Iterable[ChatCompletionChunk], response),
            request,
        )

    @staticmethod
    def _consume_stream(
        stream: Iterable[ChatCompletionChunk],
        request: ProviderRequest,
    ) -> ChatCompletion:
        state = ChatCompletionStreamState()
        received_chunk = False

        try:
            start_event = ProviderStreamEventType.START

            request.execute_on_stream_event(ProviderStreamEvent(start_event))
            for chunk in stream:
                received_chunk = True
                state.handle_chunk(chunk)
                for event in FromOpenAIChatCompletionStreamMapper.map_chunk(chunk):
                    request.execute_on_stream_event(event)
        finally:
            close = getattr(stream, "close", None)
            if callable(close):
                close()

        if not received_chunk:
            raise RuntimeError("No chunks returned from the provider")

        response = cast(ChatCompletion, state.get_final_completion())
        end_event = ProviderStreamEventType.END
        request.execute_on_stream_event(ProviderStreamEvent(end_event))
        return response

    def _parse_response(
        self,
        response: ChatCompletion,
        request: ProviderRequest,
    ) -> ProviderResponse:
        if not response.choices:
            raise RuntimeError("No choices returned from the provider")

        response_content = FromOpenAIChatCompletionMapper.map_assistant_message(
            response.choices[0].message,
            tools=request.get_tools(),
        )

        usage = FromOpenAIChatCompletionMapper.extract_usage(response)
        thinking = FromOpenAIChatCompletionMapper.extract_thinking(response)

        return ProviderResponse(
            response=response_content, thinking=thinking, usage=usage
        )

    def _parse_structured_response[ResponseModel: BaseModel](
        self,
        response: ChatCompletion,
        request: StructuredProviderRequest[ResponseModel],
    ) -> StructuredProviderResponse[ResponseModel]:
        if not response.choices:
            raise RuntimeError("No choices returned from the provider")

        provider_message = response.choices[0].message
        json_data = provider_message.content or "{}"
        response_model = request.get_response_model()
        content = response_model.model_validate_json(
            json_data,
            strict=request.get_strict(),
        )
        response_content = StructuredAssistantMessage(
            content,
            tool_calls=FromOpenAIChatCompletionMapper.map_tool_calls(
                provider_message.tool_calls,
                tools=request.get_tools(),
            ),
        )

        usage = FromOpenAIChatCompletionMapper.extract_usage(response)
        thinking = FromOpenAIChatCompletionMapper.extract_thinking(response)

        return StructuredProviderResponse(
            response=response_content,
            thinking=thinking,
            usage=usage,
        )
