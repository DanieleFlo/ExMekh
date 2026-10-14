from __future__ import annotations

from exmekh_lib.config.model_text import MODEL_TEXT
from exmekh_lib.interfaces.message import Message


class SystemMessage(Message[str]):
    def __init__(self, content: str, name: str = "") -> None:
        self._content = content
        self._role = "system"
        self._name = name
        self._check_inputs()

    def _check_inputs(self):
        if not isinstance(self._content, str):
            raise TypeError(MODEL_TEXT.system_content_must_be_a_string)
        if not isinstance(self._name, str):
            raise TypeError(MODEL_TEXT.system_name_must_be_a_string)
        if not self._content:
            raise TypeError(MODEL_TEXT.the_content_must_not_be_empty)

    def get_message(self) -> str:
        return self._content

    def get_role_name(self) -> str:
        return self._role

    def get_name(self) -> str:
        return self._name

    def message_dump(self) -> dict[str, str]:
        return {"role": self._role, "content": self._content}
