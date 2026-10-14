from __future__ import annotations

import logging
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

from exmekh_lib.config.model_text import MODEL_TEXT
from exmekh_lib.core.models.assistant_message import AssistantMessage, ToolCall
from exmekh_lib.core.models.tool_message import ToolMessage
from exmekh_lib.interfaces.message import Message

logger = logging.getLogger(__name__)

type HistoryMessage = Message[object]


@dataclass(frozen=True, slots=True)
class ToolBatchHistory:
    opening_index: int
    preceding_responses: tuple[ToolMessage, ...]


class MessageHistory:
    def __init__(self, history: Sequence[HistoryMessage] | None = None):
        if history is not None:
            self._history = list(history)
        else:
            self._history = []

    def check_history(self) -> None:
        if not self._history:
            raise ValueError(MODEL_TEXT.history_cannot_be_empty)
        if not all(isinstance(message, Message) for message in self._history):
            raise TypeError(MODEL_TEXT.item_must_be_an_instance_of_message)
        if self._history[0].get_role_name() != "system":
            raise TypeError(MODEL_TEXT.first_item_in_history_must_have_the_system_role)

        for previous, current in zip(
            self._history,
            self._history[1:],
            strict=False,
        ):
            if not self._is_valid_transition(previous, current):
                logger.warning(
                    MODEL_TEXT.wrong_message_sequence.format(
                        previous_get_role_name=previous.get_role_name(),
                        current_get_role_name=current.get_role_name(),
                    )
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
            raise TypeError(MODEL_TEXT.item_must_be_an_instance_of_message_or_a)

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

    def get_tool_continuation_messages(self) -> list[HistoryMessage]:
        """Return history only after a tool reply has completed the transition."""
        if not self._history or not isinstance(self._history[-1], ToolMessage):
            raise TypeError(MODEL_TEXT.the_last_message_must_be_a_tool_response)
        return self.get_history()

    def get_latest_tool_batch(self) -> ToolBatchHistory:
        """Locate the current batch and its already recorded sibling replies."""
        opening_index, opening = self._latest_assistant_message()
        if not opening.get_tool_calls():
            raise ValueError(MODEL_TEXT.the_latest_assistant_message_has_no_tool_calls)
        return ToolBatchHistory(
            opening_index=opening_index,
            preceding_responses=tuple(
                message
                for message in self._history[opening_index + 1 :]
                if isinstance(message, ToolMessage)
            ),
        )

    def add_tool_calls(self, tool_calls: Sequence[ToolCall]) -> None:
        self.add(AssistantMessage(None, tool_calls=list(tool_calls)))

    def replace_latest_tool_calls(self, tool_calls: Sequence[ToolCall]) -> None:
        if not self._history or not isinstance(self._history[-1], AssistantMessage):
            raise TypeError(MODEL_TEXT.the_last_message_must_be_an_assistant_message)
        self.replace_from(
            len(self._history) - 1,
            [self._history[-1].with_tool_calls(tool_calls)],
        )

    @contextmanager
    def record_tool_call(
        self, call: ToolCall, *, batch_call_id: str | None = None
    ) -> Generator[None]:
        """Record a harness call and keep an existing batch's final reply last."""
        if batch_call_id is None:
            if (
                self._history
                and isinstance(self._history[-1], AssistantMessage)
                and not self._history[-1].get_tool_calls()
            ):
                self.replace_latest_tool_calls([call])
            else:
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
            raise TypeError(MODEL_TEXT.the_last_message_must_be_a_tool_response)
        final_reply = self._history[-1]
        if final_reply.get_tool_call_id() != batch_call_id:
            raise ValueError(MODEL_TEXT.the_final_tool_response_does_not_belong_to_this)
        opening_index, opening = self._latest_assistant_message()
        calls = opening.get_tool_calls() or ()
        if not any(item.get_id() == batch_call_id for item in calls):
            raise ValueError(
                MODEL_TEXT.the_assistant_message_does_not_contain_the_batch_call
            )
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
        raise ValueError(MODEL_TEXT.the_tool_batch_has_no_assistant_message)

    def replace_from(
        self, start_index: int, messages: Sequence[HistoryMessage]
    ) -> None:
        if not isinstance(start_index, int) or isinstance(start_index, bool):
            raise TypeError(MODEL_TEXT.start_index_must_be_an_integer)
        if start_index < 0 or start_index > len(self._history):
            raise IndexError(MODEL_TEXT.start_index_is_outside_the_history)

        previous_size = len(self._history)
        candidate = MessageHistory([*self._history[:start_index], *messages])
        candidate.check_history()
        self._history = candidate.get_history()
        logger.info(
            MODEL_TEXT.compacted_message_history_start_index_previous_size_replacement_size,
            start_index,
            previous_size,
            len(messages),
            len(self._history),
        )

    def dump_history(self) -> list[dict[str, str]]:
        return [message.message_dump() for message in self._history]
