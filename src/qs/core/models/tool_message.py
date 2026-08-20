from __future__ import annotations

from qs.interfaces.message import Message


class ToolMessage(Message):
    def __init__(
        self,
        content: str,
        tool_call_id: str,
        role: str = "tool",
        name: str = "",
    ) -> None:
        self._content = content
        self._tool_call_id = tool_call_id
        self._role = role
        self._name = name

    def get_message(self) -> str:
        return self._content

    def get_role_name(self) -> str:
        return self._role

    def get_name(self) -> str:
        return self._name

    def get_tool_call_id(self) -> str:
        return self._tool_call_id

    def message_dump(self) -> dict:
        return {"role": self._role, "content": self._content}
