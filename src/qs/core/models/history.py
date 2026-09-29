from __future__ import annotations

import logging
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import cast

from qs.core.models.assistant_message import AssistantMessage, ToolCall
from qs.core.models.tool_message import ToolMessage
from qs.core.models.user_message import ErrorUserMessage, UserMessage
from qs.interfaces.message import Message

logger = logging.getLogger(__name__)

type HistoryMessage = Message[object]


class MessageHistory:
    def __init__(self, history: Sequence[HistoryMessage] | None = None):
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

        for previous, current in zip(
            self._history,
            self._history[1:],
            strict=False,
        ):
            if not self._is_valid_transition(previous, current):
                logger.warning(
                    "Wrong message sequence: "
                    f"{previous.get_role_name()} -> {current.get_role_name()}"
                )

    def _is_valid_transition(
        self, previous: HistoryMessage, current: HistoryMessage
    ) -> bool:
        previous_role = previous.get_role_name()
        current_role = current.get_role_name()

        if previous_role == "user":
            return current_role in {"assistant", "user"}

        if previous_role in {"tool", "function"}:
            # A single assistant turn may request multiple tools. Their responses
            # are consecutive messages before the next assistant continuation.
            return current_role in {"assistant", "tool", "function", "user"}

        if previous_role == "assistant":
            return current_role in {"user", "tool", "function"}

        if previous_role == "system":
            return current_role == "user"

        return True

    def add(self, message: HistoryMessage | list[HistoryMessage]) -> MessageHistory:
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

    def _add_single_message(self, message: HistoryMessage) -> MessageHistory:
        candidate = MessageHistory([*self._history, message])
        candidate.check_history()
        self._history = candidate.get_history()
        return MessageHistory(self._history)

    def _add_multiple_messages(self, messages: list[HistoryMessage]) -> MessageHistory:
        candidate = MessageHistory([*self._history, *messages])
        candidate.check_history()
        self._history = candidate.get_history()
        return MessageHistory(self._history)

    def get_history(self) -> list[HistoryMessage]:
        return list(self._history)

    def get_copy_of_history(self) -> list[HistoryMessage]:
        return [message.copy() for message in self._history]

    def add_tool_calls(self, tool_calls: Sequence[ToolCall]) -> None:
        self.add(AssistantMessage(None, tool_calls=list(tool_calls)))

    def replace_latest_tool_calls(self, tool_calls: Sequence[ToolCall]) -> None:
        if not self._history or not isinstance(self._history[-1], AssistantMessage):
            raise TypeError("The last message must be an assistant message")
        self.replace_from(
            len(self._history) - 1,
            [self._history[-1].with_tool_calls(tool_calls)],
        )

    @contextmanager
    def record_tool_call(
        self, call: ToolCall, *, batch_call_id: str | None = None
    ) -> Iterator[None]:
        """Record a harness call and keep an existing batch's final reply last."""
        if batch_call_id is None:
            self.add_tool_calls([call])
            yield
            return

        final_reply = self._extend_tool_batch(call, batch_call_id)
        try:
            yield
        finally:
            self.add(final_reply)

    def _extend_tool_batch(self, call: ToolCall, batch_call_id: str) -> ToolMessage:
        if not self._history or not isinstance(self._history[-1], ToolMessage):
            raise TypeError("The last message must be a tool response")
        final_reply = self._history[-1]
        if final_reply.get_tool_call_id() != batch_call_id:
            raise ValueError("The final tool response does not belong to this batch")
        opening_index, opening = self._latest_assistant_message()
        calls = opening.get_tool_calls() or ()
        if not any(item.id == batch_call_id for item in calls):
            raise ValueError("The assistant message does not contain the batch call")
        self.replace_from(
            opening_index,
            [
                opening.with_tool_calls([*calls, call]),
                *self._history[opening_index + 1 : -1],
            ],
        )
        return final_reply

    def _latest_assistant_message(self) -> tuple[int, AssistantMessage]:
        for index in reversed(range(len(self._history))):
            message = self._history[index]
            if isinstance(message, AssistantMessage):
                return index, message
        raise ValueError("The tool batch has no assistant message")

    def replace_from(
        self, start_index: int, messages: Sequence[HistoryMessage]
    ) -> None:
        if not isinstance(start_index, int) or isinstance(start_index, bool):
            raise TypeError("start_index must be an integer")
        if start_index < 0 or start_index > len(self._history):
            raise IndexError("start_index is outside the history")

        previous_size = len(self._history)
        candidate = MessageHistory([*self._history[:start_index], *messages])
        candidate.check_history()
        self._history = candidate.get_history()
        logger.info(
            "Compacted message history: start_index=%d previous_size=%d "
            "replacement_size=%d new_size=%d",
            start_index,
            previous_size,
            len(messages),
            len(self._history),
        )

    def compact_from(
        self, start_index: int, messages: Sequence[HistoryMessage]
    ) -> tuple[UserMessage, ...]:
        """Replace a skill transcript and return external input for parent replay."""
        previous = self._history
        self.replace_from(start_index, messages)
        return tuple(
            cast(UserMessage, message.copy())
            for message in previous[start_index:]
            if isinstance(message, UserMessage)
            and not isinstance(message, ErrorUserMessage)
        )

    def dump_history(self) -> list[dict[str, str]]:
        return [message.message_dump() for message in self._history]
