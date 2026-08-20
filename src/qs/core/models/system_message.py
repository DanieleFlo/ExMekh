from __future__ import annotations

from qs.interfaces.message import Message


class SystemMessage(Message):
    def __init__(self, content: str, role: str = "system", name: str = "") -> None:
        self._content = content
        self._role = role
        self._name = name

    def get_message(self) -> str:
        return self._content

    def get_role_name(self) -> str:
        return self._role

    def get_name(self) -> str:
        return self._name

    def message_dump(self) -> dict:
        return {"role": self._role, "content": self._content}
