from __future__ import annotations

from typing import ClassVar, List, Optional

from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionDeveloperMessageParam,
    ChatCompletionFunctionMessageParam,
    ChatCompletionMessage,
    ChatCompletionMessageFunctionToolCallParam,
    ChatCompletionMessageToolCallUnion,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_content_part_param import (
    File,
    FileFile,
)
from openai.types.chat.chat_completion_message_custom_tool_call import ChatCompletionMessageCustomToolCall
from openai.types.chat.chat_completion_message_custom_tool_call_param import (
    ChatCompletionMessageCustomToolCallParam,
    Custom as OpenAICustom,
)
from openai.types.chat.chat_completion_message_function_tool_call import ChatCompletionMessageFunctionToolCall
from openai.types.chat.chat_completion_message_function_tool_call_param import Function as OpenAIFunction
from openai.types.chat.chat_completion_message_tool_call_union_param import ChatCompletionMessageToolCallUnionParam

from qs.core.models.assistant_message import (
    AssistantMessage,
    CustomCall,
    CustomToolCall,
    FunctionCall,
    FunctionToolCall,
    ToolCall,
)
from qs.core.models.function_message import FunctionMessage
from qs.core.models.model_thinking import ModelThinking
from qs.core.models.model_usage import ModelUsage
from qs.core.models.system_message import SystemMessage
from qs.core.models.tool_message import ToolMessage
from qs.core.models.user_message import (
    DocumentUserMessage,
    FilesUserMessage,
    ImagesUserMessage,
    TextsUserMessage,
    UserMessage,
)
from qs.utils.file import convert_bytes_to_base64
from qs.utils.image import pil_image_to_data_url


class OpenAIChatCompletionMapper:
    _THINKING_FIELDS: ClassVar[tuple[str, ...]] = (
        "reasoning",
        "reasoning_content",
        "thinking",
    )

    # =========================
    # Public mapping interface
    # =========================

    @staticmethod
    def extract_usage(response: ChatCompletion) -> ModelUsage:
        usage = response.usage

        if usage is None:
            return ModelUsage(
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
            )

        return ModelUsage(
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
        )

    @classmethod
    def extract_thinking(
        cls,
        response: ChatCompletion,
    ) -> ModelThinking | None:
        for choice in response.choices:
            thinking = cls._thinking_from(choice.message)

            if thinking is not None:
                return ModelThinking(thinking)

        return None

    @staticmethod
    def map_system_message(
        system_message: SystemMessage,
    ) -> ChatCompletionSystemMessageParam | ChatCompletionDeveloperMessageParam:
        role = system_message.get_role_name()
        content = system_message.get_message()
        name = system_message.get_name()

        if role == "system":
            return ChatCompletionSystemMessageParam(
                content=content, role=role, name=name
            )

        if role == "developer":
            return ChatCompletionDeveloperMessageParam(
                content=content, role=role, name=name
            )

        raise ValueError(f"Expected role 'system' or 'developer', got '{role}'")

    @classmethod
    def map_user_message(
        cls,
        user_message: UserMessage,
    ) -> ChatCompletionUserMessageParam:
        role = user_message.get_role_name()
        content = user_message.get_message()

        if role != "user":
            raise ValueError(f"Expected role 'user', got '{role}'")

        return ChatCompletionUserMessageParam(
            content=cls._map_user_content(content),
            role=role,
            name=user_message.get_name(),
        )

    @classmethod
    def map_assistant_message(
        cls,
        assistant_message: AssistantMessage,
    ) -> ChatCompletionAssistantMessageParam:
        role = assistant_message.get_role_name()
        content = assistant_message.get_message()

        if role != "assistant":
            raise ValueError(f"Expected role 'assistant', got '{role}'")

        mapped_message = ChatCompletionAssistantMessageParam(
            content=content,
            role=role,
            name=assistant_message.get_name(),
        )

        tool_calls = assistant_message.get_tool_calls()
        if tool_calls is not None:
            mapped_message["tool_calls"] = [
                cls._map_tool_call(tool_call) for tool_call in tool_calls
            ]

        return mapped_message

    @staticmethod
    def map_tool_message(
        tool_message: ToolMessage,
    ) -> ChatCompletionToolMessageParam:
        role = tool_message.get_role_name()
        content = tool_message.get_message()

        if role != "tool":
            raise ValueError(f"Expected role 'tool', got '{role}'")

        return ChatCompletionToolMessageParam(
            content=content,
            role=role,
            tool_call_id=tool_message.get_tool_call_id(),
        )

    @staticmethod
    def map_function_message(
        function_message: FunctionMessage,
    ) -> ChatCompletionFunctionMessageParam:
        role = function_message.get_role_name()
        content = function_message.get_message()

        if role != "function":
            raise ValueError(f"Expected role 'function', got '{role}'")

        return ChatCompletionFunctionMessageParam(
            content=content,
            role=role,
            name=function_message.get_name(),
        )

    @staticmethod
    def map_tool_calls(
        tool_calls: List[ChatCompletionMessageToolCallUnion] | None,
    ) -> List[ToolCall] | None:
        if tool_calls is None:
            return None

        mapped_tool_calls: List[ToolCall] = []

        for tool_call in tool_calls:
            if isinstance(tool_call, ChatCompletionMessageFunctionToolCall):
                mapped_tool_calls.append(
                    FunctionToolCall(
                        id=tool_call.id,
                        function=FunctionCall(
                            name=tool_call.function.name,
                            arguments=tool_call.function.arguments,
                        ),
                        type="function",
                    )
                )
                continue

            if isinstance(tool_call, ChatCompletionMessageCustomToolCall):
                mapped_tool_calls.append(
                    CustomToolCall(
                        id=tool_call.id,
                        custom=CustomCall(
                            name=tool_call.custom.name,
                            input=tool_call.custom.input,
                        ),
                        type="custom",
                    )
                )
                continue

            raise TypeError(
                f"Unsupported OpenAI tool call type: {type(tool_call).__name__}"
            )

        return mapped_tool_calls

    # =========================
    # Internal implementation
    # =========================

    @classmethod
    def _map_tool_call(
        cls,
        tool_call: ToolCall,
    ) -> ChatCompletionMessageToolCallUnionParam:
        if tool_call["type"] == "function":
            return cls._map_function_tool_call(tool_call)

        if tool_call["type"] == "custom":
            return cls._map_custom_tool_call(tool_call)

        raise ValueError(f"Unsupported tool call type: {tool_call['type']!r}")

    @staticmethod
    def _map_function_tool_call(
        tool_call: FunctionToolCall,
    ) -> ChatCompletionMessageFunctionToolCallParam:
        function = tool_call["function"]
        return ChatCompletionMessageFunctionToolCallParam(
            id=tool_call["id"],
            function=OpenAIFunction(
                name=function["name"],
                arguments=function["arguments"],
            ),
            type="function",
        )

    @staticmethod
    def _map_custom_tool_call(
        tool_call: CustomToolCall,
    ) -> ChatCompletionMessageCustomToolCallParam:
        custom = tool_call["custom"]
        return ChatCompletionMessageCustomToolCallParam(
            id=tool_call["id"],
            custom=OpenAICustom(
                name=custom["name"],
                input=custom["input"],
            ),
            type="custom",
        )

    @classmethod
    def _map_user_content(
        cls,
        content: str | TextsUserMessage | ImagesUserMessage | FilesUserMessage,
    ) -> str | List[ChatCompletionContentPartParam]:

        if isinstance(content, str):
            return content

        if isinstance(content, TextsUserMessage):
            return cls._map_text_parts(content)

        if isinstance(content, ImagesUserMessage):
            return cls._map_image_parts(content)

        if isinstance(content, FilesUserMessage):
            return cls._map_file_parts(content)

        raise TypeError(f"Unsupported user content type: {type(content).__name__}")

    @staticmethod
    def _map_text_parts(
        content: TextsUserMessage,
    ) -> list[ChatCompletionContentPartParam]:
        return [
            ChatCompletionContentPartTextParam(
                text=text,
                type="text",
            )
            for text in content.texts
        ]

    @staticmethod
    def _map_image_parts(
        content: ImagesUserMessage,
    ) -> list[ChatCompletionContentPartParam]:
        return [
            ChatCompletionContentPartImageParam(
                image_url={"url": pil_image_to_data_url(image)},
                type="image_url",
            )
            for image in content.images
        ]

    @classmethod
    def _map_file_parts(
        cls,
        content: FilesUserMessage,
    ) -> list[ChatCompletionContentPartParam]:
        return [
            cls._map_file_document(document, document_index)
            for document_index, document in enumerate(content.documents)
        ]

    @staticmethod
    def _map_file_document(
        document: DocumentUserMessage,
        document_index: int,
    ) -> File:
        filename = f"{document.name}_{document_index}.{document.extension}"
        file_in_base64 = convert_bytes_to_base64(document.file)

        file_data: FileFile = {
            "filename": filename,
            "file_data": f"data:{document.mime_type};base64,{file_in_base64}",
        }

        return File(
            type="file",
            file=file_data,
        )

    @classmethod
    def _thinking_from(
        cls,
        message: ChatCompletionMessage,
    ) -> str | None:
        for field in cls._THINKING_FIELDS:
            value = getattr(message, field, None)

            if isinstance(value, str) and value.strip():
                return value

        return None
