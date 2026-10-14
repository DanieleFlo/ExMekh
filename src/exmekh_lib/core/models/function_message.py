from __future__ import annotations

from exmekh_lib.config.model_text import MODEL_TEXT
from exmekh_lib.interfaces.message import Message


class FunctionMessage(Message[str | None]):
    def __init__(
        self,
        content: str | None,
        name: str,
    ) -> None:
        self._content = content
        self._role: str = "function"
        self._name = name
        self._check_inputs()

    def _check_inputs(self):
        if self._content is not None and not isinstance(self._content, str):
            raise TypeError(MODEL_TEXT.function_content_must_be_a_string)
        if not isinstance(self._name, str):
            raise TypeError(MODEL_TEXT.function_name_must_be_a_string)

    def get_message(self) -> str | None:
        return self._content

    def get_role_name(self) -> str:
        return self._role

    def get_name(self) -> str:
        return self._name

    def message_dump(self) -> dict[str, str]:
        content = self._content if self._content is not None else ""
        return {"role": self._role, "content": content}
