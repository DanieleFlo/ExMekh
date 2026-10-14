from __future__ import annotations

from exmekh_lib.config.model_text import MODEL_TEXT
from exmekh_lib.core.models.tool import PreformattedToolResult
from exmekh_lib.core.models.user_message import UserMessage
from exmekh_lib.interfaces.message import Message


class ToolMessage(Message[str]):
    def with_result(self, result: PreformattedToolResult) -> ToolMessage:
        return ToolMessage(
            result.content,
            self._tool_call_id,
            self._name,
            attachments=result.attachments,
        )

    def __init__(
        self,
        content: str,
        tool_call_id: str,
        name: str = "",
        *,
        attachments: UserMessage | None = None,
    ) -> None:
        self._content = content
        self._tool_call_id = tool_call_id
        self._role: str = "tool"
        self._name = name
        self._attachments = attachments
        self._check_inputs()

    def _check_inputs(self):
        if not isinstance(self._content, str):
            raise TypeError(MODEL_TEXT.tool_content_must_be_a_string)
        if not isinstance(self._tool_call_id, str):
            raise TypeError(MODEL_TEXT.tool_call_id_must_be_a_string)
        if not isinstance(self._name, str):
            raise TypeError(MODEL_TEXT.tool_name_must_be_a_string)
        if self._attachments is not None and not isinstance(
            self._attachments, UserMessage
        ):
            raise TypeError(MODEL_TEXT.tool_attachments_must_be_a_usermessage_or_none)

    def get_message(self) -> str:
        return self._content

    def get_role_name(self) -> str:
        return self._role

    def get_name(self) -> str:
        return self._name

    def get_tool_call_id(self) -> str:
        return self._tool_call_id

    def get_attachments(self) -> UserMessage | None:
        return self._attachments

    def message_dump(self) -> dict[str, str]:
        return {"role": self._role, "content": self._content}
