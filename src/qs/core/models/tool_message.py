from __future__ import annotations

from qs.core.models.user_message import UserMessage
from qs.interfaces.message import Message


class ToolMessage(Message[str]):
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
            raise TypeError("Tool content must be a string")
        if not isinstance(self._tool_call_id, str):
            raise TypeError("Tool call ID must be a string")
        if not isinstance(self._name, str):
            raise TypeError("Tool name must be a string")
        if self._attachments is not None and not isinstance(
            self._attachments, UserMessage
        ):
            raise TypeError("Tool attachments must be a UserMessage or None")

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
