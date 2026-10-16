"""Find replaceable spans without splitting tool batches or protected context."""

from collections.abc import Sequence
from dataclasses import dataclass

from exmekh_lib.config.history_compaction import HISTORY_COMPACTION
from exmekh_lib.core.models.assistant_message import AssistantMessage
from exmekh_lib.core.models.history import HistoryMessage
from exmekh_lib.core.models.structured_assistant_message import (
    StructuredAssistantMessage,
)
from exmekh_lib.core.models.tool_message import ToolMessage
from exmekh_lib.core.models.xml_header import XmlHeader


@dataclass(frozen=True, slots=True)
class HistorySpan:
    start: int
    end: int


def complete_batches(messages: Sequence[HistoryMessage]) -> tuple[HistorySpan, ...]:
    pending: dict[str, str] = {}
    batches: list[HistorySpan] = []
    start = 0
    for index, message in enumerate(messages):
        if isinstance(message, ToolMessage):
            if pending.pop(message.get_tool_call_id(), None) != message.get_name():
                raise ValueError(HISTORY_COMPACTION.invalid_history)
            if not pending:
                batches.append(HistorySpan(start, index + 1))
            continue
        if pending:
            raise ValueError(HISTORY_COMPACTION.invalid_history)
        if isinstance(message, (AssistantMessage, StructuredAssistantMessage)):
            calls = message.get_tool_calls() or ()
            pending = {call.id: call.get_name() for call in calls}
            if len(pending) != len(calls):
                raise ValueError(HISTORY_COMPACTION.invalid_history)
            start = index
    if pending:
        raise ValueError(HISTORY_COMPACTION.invalid_history)
    return tuple(batches)


def compaction_spans(
    messages: Sequence[HistoryMessage], protected_indices: Sequence[int]
) -> tuple[HistorySpan, ...]:
    protected = set(protected_indices)
    if any(
        type(index) is not int or not 0 <= index < len(messages) for index in protected
    ):
        raise ValueError(HISTORY_COMPACTION.invalid_anchor)
    protected.update(
        index
        for index, message in enumerate(messages)
        if message.get_role_name() in {"system", "user"}
    )
    for batch in complete_batches(messages):
        already_compacted = any(
            isinstance(message, ToolMessage)
            and message.get_name() == HISTORY_COMPACTION.tool_name
            and XmlHeader.field_from_content(message.get_message(), "compacted")
            == "true"
            for message in messages[batch.start : batch.end]
        )
        if already_compacted or any(
            index in protected for index in range(batch.start, batch.end)
        ):
            protected.update(range(batch.start, batch.end))
    spans: list[HistorySpan] = []
    start = 0
    for index in range(len(messages) + 1):
        if index == len(messages) or index in protected:
            if start < index:
                spans.append(HistorySpan(start, index))
            start = index + 1
    return tuple(spans)
