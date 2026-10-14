from __future__ import annotations

from pydantic import BaseModel

from exmekh_lib.config.model_text import MODEL_TEXT
from exmekh_lib.core.models.assistant_message import ToolCall
from exmekh_lib.interfaces.message import Message


class StructuredAssistantMessage[MessageContent: BaseModel](Message[MessageContent]):
    def __init__(
        self,
        content: MessageContent,
        name: str = "",
        tool_calls: list[ToolCall] | None = None,
    ) -> None:
        self._content = content
        self._role = "assistant"
        self._name = name
        self._tool_calls = tool_calls
        self._check_inputs()

    def _check_inputs(self):
        if not issubclass(type(self._content), BaseModel):
            raise TypeError(
                MODEL_TEXT.structured_assistant_content_must_be_a_pydantic_basemodel
            )
        if self._tool_calls is not None:
            if not isinstance(self._tool_calls, list):
                raise TypeError(
                    MODEL_TEXT.structured_assistant_tool_calls_must_be_a_list_of
                )
            for tool_call in self._tool_calls:
                if not isinstance(tool_call, ToolCall):
                    raise TypeError(
                        MODEL_TEXT.structured_assistant_tool_calls_must_be_a_list_of
                    )

    def get_message(self) -> MessageContent:
        if not issubclass(type(self._content), BaseModel):
            raise TypeError(
                MODEL_TEXT.content_must_be_a_subclass_of_basemodel_got.format(
                    content_type=type(self._content)
                )
            )
        return self._content

    def get_role_name(self) -> str:
        return self._role

    def get_name(self) -> str:
        return self._name

    def get_tool_calls(self) -> list[ToolCall] | None:
        return self._tool_calls

    def message_dump(self) -> dict[str, str]:
        _tool_calls = self._tool_calls
        if _tool_calls is None:
            tool_calls_content = "null"
        else:
            tool_calls_content = ToolCall.dump_many(_tool_calls)
        return {
            "role": self._role,
            "content": self._content.model_dump_json(),
            "tool_calls": tool_calls_content,
        }
