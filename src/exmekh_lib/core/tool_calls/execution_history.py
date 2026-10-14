from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from exmekh_lib.config.runtime import RUNTIME
from exmekh_lib.config.runtime_text import RUNTIME_TEXT
from exmekh_lib.core.cache.node_cache import NodeCache
from exmekh_lib.core.cache.tool_call_record import ToolCallRecord
from exmekh_lib.utils.tool_errors import ToolInputError


@dataclass(slots=True)
class ExecutionHistoryEntry:
    position: int
    name: str
    call_id: str
    saved_output_id: str | None = None
    saved_session_dir: Path | None = None

    def attach_output(self, output_id: str, session_dir: Path) -> None:
        self.saved_output_id = output_id
        self.saved_session_dir = session_dir

    def recoverable_output_id(self, cache: NodeCache) -> str | None:
        if (
            self.saved_output_id is not None
            and self.saved_session_dir == cache.session_dir
            and cache.has(self.saved_output_id)
        ):
            return self.saved_output_id
        return None

    def read_output(self, cache: NodeCache) -> ToolCallRecord | None:
        if self.saved_output_id is None or self.saved_session_dir != cache.session_dir:
            return None
        return cache.get(self.saved_output_id)


@dataclass(frozen=True, slots=True)
class OmittedHistoryRange:
    start_position: int
    end_position: int
    count: int


@dataclass(frozen=True, slots=True)
class ExecutionHistoryView:
    mode: Literal["overview", "page"]
    total: int
    entries: tuple[ExecutionHistoryEntry, ...]
    omitted: OmittedHistoryRange | None = None
    start_position: int | None = None
    end_position: int | None = None
    previous_position: int | None = None
    next_position: int | None = None


class ExecutionHistory:
    """Compact capability-call history retained for one node runtime."""

    def __init__(self) -> None:
        self._entries: list[ExecutionHistoryEntry] = []

    def record(self, name: str, call_id: str) -> ExecutionHistoryEntry:
        entry = ExecutionHistoryEntry(len(self._entries) + 1, name, call_id)
        self._entries.append(entry)
        return entry

    def attach_output(
        self,
        call_id: str,
        output_id: str,
        session_dir: Path,
    ) -> None:
        for entry in reversed(self._entries):
            if entry.call_id == call_id:
                entry.attach_output(output_id, session_dir)
                return

    def attach_output_at_position(
        self, position: int, output_id: str, session_dir: Path
    ) -> None:
        if not 1 <= position <= len(self._entries):
            raise ValueError(RUNTIME_TEXT.execution_history_position_is_out_of_range)
        self._entries[position - 1].attach_output(output_id, session_dir)

    @property
    def entries(self) -> tuple[ExecutionHistoryEntry, ...]:
        return tuple(self._entries)

    @property
    def count(self) -> int:
        return len(self._entries)

    def find_by_identifier(self, identifier: str) -> tuple[ExecutionHistoryEntry, ...]:
        return tuple(
            entry
            for entry in self._entries
            if identifier in (entry.call_id, entry.saved_output_id)
        )

    def select(
        self,
        *,
        start_position: int | None = None,
        limit: int = RUNTIME.memory_default_page_limit,
    ) -> ExecutionHistoryView:
        self._validate_navigation(start_position, limit)
        if start_position is None:
            return self._overview(limit)
        return self._page(start_position, limit)

    def _overview(self, limit: int) -> ExecutionHistoryView:
        entries = self.entries
        total = len(entries)
        if total <= limit:
            return ExecutionHistoryView("overview", total, entries)

        head_count = max(1, limit // RUNTIME.history_head_divisor)
        tail_count = limit - head_count
        omitted_start = head_count + 1
        omitted_end = total - tail_count
        return ExecutionHistoryView(
            mode="overview",
            total=total,
            entries=(*entries[:head_count], *entries[-tail_count:]),
            omitted=OmittedHistoryRange(
                start_position=omitted_start,
                end_position=omitted_end,
                count=omitted_end - omitted_start + 1,
            ),
        )

    def _page(self, start_position: int, limit: int) -> ExecutionHistoryView:
        total = len(self._entries)
        page = tuple(self._entries[start_position - 1 : start_position - 1 + limit])
        if not page:
            previous_position = None
            if total:
                previous_position = ((total - 1) // limit) * limit + 1
            return ExecutionHistoryView(
                mode="page",
                total=total,
                entries=(),
                previous_position=previous_position,
            )

        actual_start = page[0].position
        actual_end = page[-1].position
        return ExecutionHistoryView(
            mode="page",
            total=total,
            entries=page,
            start_position=actual_start,
            end_position=actual_end,
            previous_position=max(1, actual_start - limit)
            if actual_start > 1
            else None,
            next_position=actual_end + 1 if actual_end < total else None,
        )

    @staticmethod
    def _validate_navigation(start_position: int | None, limit: int) -> None:
        if start_position is not None and (
            type(start_position) is not int or start_position < 1
        ):
            raise ToolInputError(
                RUNTIME_TEXT.start_position_must_be_a_positive_integer_or_none,
                code="invalid_history_page",
            )
        if (
            type(limit) is not int
            or not RUNTIME.history_min_page_limit
            <= limit
            <= RUNTIME.memory_max_page_limit
        ):
            raise ToolInputError(
                RUNTIME_TEXT.limit_must_be_between_and.format(
                    runtime_history_min_page_limit=RUNTIME.history_min_page_limit,
                    runtime_memory_max_page_limit=RUNTIME.memory_max_page_limit,
                ),
                code="invalid_history_page",
            )
