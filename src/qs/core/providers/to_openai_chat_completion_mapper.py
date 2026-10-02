from __future__ import annotations

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionDeveloperMessageParam,
    ChatCompletionFunctionMessageParam,
    ChatCompletionMessageFunctionToolCallParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_content_part_param import (
    File,
    FileFile,
)
from openai.types.chat.chat_completion_message_function_tool_call_param import (
    Function as OpenAIFunction,
)
from openai.types.chat.chat_completion_message_tool_call_union_param import (
    ChatCompletionMessageToolCallUnionParam,
)
from PIL import Image
from pydantic import BaseModel
from pydantic_core import to_json

from qs.core.models.assistant_message import (
    AssistantMessage,
    ToolCall,
)
from qs.core.models.function_message import FunctionMessage
from qs.core.models.structured_assistant_message import StructuredAssistantMessage
from qs.core.models.system_message import SystemMessage
from qs.core.models.tool_message import ToolMessage
from qs.core.models.user_message import (
    DocumentUserMessage,
    UserMessage,
)
from qs.utils.file import convert_bytes_to_base64
from qs.utils.image import pil_image_to_data_url


class ToOpenAIChatCompletionMapper:
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

        if role != "user":
            raise ValueError(f"Expected role 'user', got '{role}'")

        return ChatCompletionUserMessageParam(
            content=cls._map_user_content(user_message),
            role=role,
            name=user_message.get_name(),
        )

    @classmethod
    def _map_user_content(
        cls,
        message: UserMessage,
    ) -> str | list[ChatCompletionContentPartParam]:
        content = message.get_message()
        if isinstance(content, str):
            return content
        parts: list[ChatCompletionContentPartParam] = []
        document_index = 0
        for part in message.iter_content_parts():
            if isinstance(part, str):
                parts.append(ChatCompletionContentPartTextParam(type="text", text=part))
            elif isinstance(part, Image.Image):
                parts.append(
                    ChatCompletionContentPartImageParam(
                        type="image_url",
                        image_url={"url": pil_image_to_data_url(part)},
                    )
                )
            elif isinstance(part, DocumentUserMessage):
                parts.append(cls._map_file_document(part, document_index))
                document_index += 1
            else:
                raise TypeError(f"Unsupported user content type: {type(part).__name__}")
        return parts

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
            mapped_message["tool_calls"] = cls._map_tool_calls(tool_calls)

        return mapped_message

    @classmethod
    def map_structured_assistant_message(
        cls,
        assistant_message: StructuredAssistantMessage[BaseModel],
    ) -> ChatCompletionAssistantMessageParam:
        role = assistant_message.get_role_name()

        if role != "assistant":
            raise ValueError(f"Expected role 'assistant', got '{role}'")

        message_content = assistant_message.get_message()

        if not isinstance(message_content, BaseModel):
            raise TypeError(
                f"Structured assistant content must be a Pydantic BaseModel, "
                f"got {type(message_content)!r}"
            )

        mapped_message = ChatCompletionAssistantMessageParam(
            content=message_content.model_dump_json(),
            role=role,
            name=assistant_message.get_name(),
        )
        tool_calls = assistant_message.get_tool_calls()

        if tool_calls is not None:
            mapped_message["tool_calls"] = cls._map_tool_calls(tool_calls)

        return mapped_message

    @classmethod
    def _map_tool_calls(
        cls,
        tool_calls: list[ToolCall],
    ) -> list[ChatCompletionMessageToolCallUnionParam]:
        return [cls._map_tool_call(tool_call) for tool_call in tool_calls]

    @classmethod
    def _map_tool_call(
        cls,
        tool_call: ToolCall,
    ) -> ChatCompletionMessageToolCallUnionParam:
        if isinstance(tool_call, ToolCall):
            return cls._map_function_tool_call(tool_call)

        raise TypeError(f"Unsupported tool call type: {type(tool_call).__name__}")

    @staticmethod
    def _map_function_tool_call(
        tool_call: ToolCall,
    ) -> ChatCompletionMessageFunctionToolCallParam:
        arguments = to_json(tool_call.get_arguments()).decode()

        return ChatCompletionMessageFunctionToolCallParam(
            id=tool_call.get_id(),
            function=OpenAIFunction(
                name=tool_call.get_name(),
                arguments=arguments,
            ),
            type="function",
        )

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
