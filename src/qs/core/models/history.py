from __future__ import annotations

from qs.core.models.system_message import SystemMessage
from qs.interfaces.message import Message

# Literal["developer", "system", "user", "assistant", "tool", "function"]


class History:
    def __init__(self, history: list[Message] | None = None):
        if history is not None:
            self._history = history
        else:
            self._history = []

    def check_history(self) -> None:
        if not all(isinstance(message, Message) for message in self._history):
            raise TypeError("Item must be an instance of Message")
        if not isinstance(self._history[0], SystemMessage):
            raise TypeError(
                "First item in history must be an instance of SystemMessage"
            )

        # TODO: Add additional checks structure history. user | assistant | tool | function
        # if len(self._history) > 1:
        #     for message in self._history[1:]:
        #         pass

    def add(self, item: Message | list[Message]) -> History:
        if isinstance(item, Message):
            return self._add_single_message(item)
        elif isinstance(item, list) and all(
            isinstance(message, Message) for message in item
        ):
            return self._add_multiple_messages(item)
        else:
            raise TypeError(
                "Item must be an instance of Message or a list of Message instances"
            )

    def _add_single_message(self, message: Message) -> History:
        self._history.append(message)
        self.check_history()
        return History(self._history)

    def _add_multiple_messages(self, messages: list[Message]) -> History:
        self._history.extend(messages)
        self.check_history()
        return History(self._history)

    def get_history(self) -> list[Message]:
        return self._history
