from __future__ import annotations

from qs.interfaces.message import Message


class SystemMessage(Message[str]):
    def __init__(self, content: str, name: str = "") -> None:
        self._content = content
        self._role = "system"
        self._name = name
        self._check_inputs()

    def _check_inputs(self):
        if not isinstance(self._content, str):
            raise TypeError("System content must be a string")
        if not isinstance(self._name, str):
            raise TypeError("System name must be a string")
        if not self._content:
            raise TypeError("The content must not be empty!")

    def get_message(self) -> str:
        return self._content

    def get_role_name(self) -> str:
        return self._role

    def get_name(self) -> str:
        return self._name

    def message_dump(self) -> dict[str, str]:
        return {"role": self._role, "content": self._content}
