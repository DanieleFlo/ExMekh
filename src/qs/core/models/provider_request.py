from __future__ import annotations

from qs.core.models.tools import Tools
from qs.interfaces.message import Message


class ProviderRequest:
    def __init__(
        self,
        messages: list[Message],
        *,
        extra: dict | None = None,
        tool_choice: str | None = None,
        tools: Tools | None = None,
    ) -> None:
        self._messages = messages
        self._extra = extra
        self._tool_choice = tool_choice
        self._tools = tools
        ProviderRequest._check_inputs(self)
        self._messages = messages.copy()
        self._extra = extra.copy() if extra is not None else None

    def _check_inputs(self) -> None:
        if not isinstance(self._messages, list) or not self._messages:
            raise ValueError("Expected a non-empty list of Messages")
        if not all(isinstance(message, Message) for message in self._messages):
            raise TypeError("All messages must be Message instances")
        if self._extra is not None and not isinstance(self._extra, dict):
            raise TypeError("Extra provider data must be a dictionary")
        if self._tool_choice is not None and not isinstance(self._tool_choice, str):
            raise TypeError("Tool choice must be a string")
        if self._tools is not None and not isinstance(self._tools, Tools):
            raise TypeError("Tools must be a Tools instance")

    def get_messages(self) -> list[Message]:
        return self._messages.copy()

    def get_extra(self) -> dict | None:
        return self._extra.copy() if self._extra is not None else None

    def get_tool_choice(self) -> str | None:
        return self._tool_choice

    def get_tools(self) -> Tools | None:
        return self._tools
