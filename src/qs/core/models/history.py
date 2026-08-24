from __future__ import annotations

import logging
from typing import List

from qs.interfaces.message import Message

logger = logging.getLogger(__name__)


class MessageHistory:
    def __init__(self, history: List[Message] | None = None):
        if history is not None:
            self._history = list(history)
        else:
            self._history = []

    def check_history(self) -> None:
        if not self._history:
            raise ValueError("History cannot be empty")
        if not all(isinstance(message, Message) for message in self._history):
            raise TypeError("Item must be an instance of Message")
        if self._history[0].get_role_name() != "system":
            raise TypeError("First item in history must have the system role")

        for previous, current in zip(self._history, self._history[1:]):
            if not self._is_valid_transition(previous, current):
                logger.warning(
                    "Wrong message sequence: "
                    f"{previous.get_role_name()} -> {current.get_role_name()}"
                )

    def _is_valid_transition(self, previous: Message, current: Message) -> bool:
        previous_role = previous.get_role_name()
        current_role = current.get_role_name()

        if previous_role == "user":
            return current_role == "assistant"

        if previous_role in {"tool", "function"}:
            return current_role == "assistant"

        if previous_role == "assistant":
            return current_role in {"user", "tool", "function"}

        if previous_role == "system":
            return current_role == "user"

        return True

    def add(self, message: Message | List[Message]) -> MessageHistory:
        if isinstance(message, Message):
            return self._add_single_message(message)
        elif (
            message
            and isinstance(message, list)
            and all(isinstance(m, Message) for m in message)
        ):
            return self._add_multiple_messages(message)
        else:
            raise TypeError(
                "Item must be an instance of Message or a list of Message instances"
            )

    def _add_single_message(self, message: Message) -> MessageHistory:
        candidate = MessageHistory([*self._history, message])
        candidate.check_history()
        self._history = candidate.get_history()
        return MessageHistory(self._history)

    def _add_multiple_messages(self, messages: list[Message]) -> MessageHistory:
        candidate = MessageHistory([*self._history, *messages])
        candidate.check_history()
        self._history = candidate.get_history()
        return MessageHistory(self._history)

    def get_history(self) -> list[Message]:
        return list(self._history)

    def dump_history(self) -> list[dict[str, str]]:
        return [message.message_dump() for message in self._history]
