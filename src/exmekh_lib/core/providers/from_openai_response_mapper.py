from __future__ import annotations

from openai.types.responses import (
    Response,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputRefusal,
    ResponseOutputText,
    ResponseReasoningItem,
)

from exmekh_lib.config.provider_text import PROVIDER_TEXT
from exmekh_lib.core.models.assistant_message import (
    AssistantMessage,
    FunctionCall,
    ToolCall,
)
from exmekh_lib.core.models.model_thinking import ModelThinking
from exmekh_lib.core.models.model_usage import ModelUsage
from exmekh_lib.core.models.tools import Tools
from exmekh_lib.core.providers.openai_response_tool_mapper import (
    OpenAIResponseToolMapper,
)


class FromOpenAIResponseMapper:
    @staticmethod
    def extract_usage(response: Response) -> ModelUsage:
        FromOpenAIResponseMapper._ensure_response(response)
        usage = response.usage
        if usage is None:
            return ModelUsage()
        return ModelUsage(
            prompt_tokens=usage.input_tokens,
            completion_tokens=usage.output_tokens,
            total_tokens=usage.total_tokens,
        )

    @staticmethod
    def extract_thinking(response: Response) -> ModelThinking | None:
        FromOpenAIResponseMapper._ensure_response(response)
        thoughts: list[str] = []

        for item in response.output:
            if not isinstance(item, ResponseReasoningItem):
                continue

            content = item.content or []
            source = content if content else item.summary
            thoughts.extend(
                text
                for part in source
                if isinstance((text := getattr(part, "text", None)), str)
                and text.strip()
            )

        if not thoughts:
            return None
        return ModelThinking("\n".join(thoughts))

    @classmethod
    def map_assistant_message(
        cls,
        response: Response,
        tools: Tools | None = None,
    ) -> AssistantMessage:
        cls._ensure_response(response)
        content: list[str] = []
        tool_calls: list[ToolCall] = []

        for item in response.output:
            if isinstance(item, ResponseOutputMessage):
                content.extend(cls._map_message_content(item))
            elif isinstance(item, ResponseFunctionToolCall):
                tool_calls.append(cls._map_tool_call(item, tools))

        return AssistantMessage(
            content="".join(content) or None,
            tool_calls=tool_calls or None,
        )

    @staticmethod
    def _map_message_content(message: ResponseOutputMessage) -> list[str]:
        content: list[str] = []
        for part in message.content:
            if isinstance(part, ResponseOutputText):
                content.append(part.text)
            elif isinstance(part, ResponseOutputRefusal):
                content.append(part.refusal)
        return content

    @staticmethod
    def _map_tool_call(
        item: ResponseFunctionToolCall,
        tools: Tools | None,
    ) -> ToolCall:
        if tools is None:
            raise ValueError(
                PROVIDER_TEXT.tools_are_required_to_resolve_a_function_tool_call
            )

        tool, arguments = OpenAIResponseToolMapper.resolve_tool_call(
            item,
            tools,
        )
        return ToolCall(
            id=item.call_id,
            function=FunctionCall(name=tool.get_name(), arguments=arguments),
        )

    @staticmethod
    def _ensure_response(response: Response) -> None:
        if not isinstance(response, Response):
            raise TypeError(PROVIDER_TEXT.expected_an_openai_response_instance)
