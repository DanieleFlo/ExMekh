from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from xml.etree import ElementTree as ET

from qs.capabilities.tools.get_execution_history.main import (
    create_tool as create_history_tool,
)
from qs.capabilities.tools.read_tool_output.main import (
    create_tool as create_read_tool,
)
from qs.capabilities.tools.search_tool_outputs.main import (
    create_tool as create_search_tool,
)
from qs.config.runtime import RUNTIME
from qs.core.cache.node_cache import NodeCache
from qs.core.cache.tool_call_record import ToolCallRecord
from qs.core.models.tool import PreformattedToolResult, Tool
from qs.utils.xml import append_text_element, render_xml

from .execution_manager import ToolExecutionManager
from .output_reader import ToolOutputReader
from .output_view import ToolOutputView
from .search import ToolCallSearch


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
            raise ValueError("execution history position is out of range")
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
            raise ValueError("start_position must be a positive integer or None")
        if (
            type(limit) is not int
            or not RUNTIME.history_min_page_limit
            <= limit
            <= RUNTIME.memory_max_page_limit
        ):
            raise ValueError(
                f"limit must be between {RUNTIME.history_min_page_limit} and {RUNTIME.memory_max_page_limit}"
            )


class HarnessMemoryTools:
    """Node-scoped XML tools for inspecting execution and cached outputs."""

    HISTORY_TOOL_NAME = RUNTIME.history_tool_name
    SEARCH_TOOL_NAME = RUNTIME.search_tool_name
    READ_TOOL_NAME = RUNTIME.read_tool_name
    PREVIEW_LIMIT = RUNTIME.memory_preview_limit

    def __init__(
        self,
        history: ExecutionHistory,
        cache_provider: Callable[[], NodeCache],
        execution_manager: ToolExecutionManager | None = None,
    ) -> None:
        self._history = history
        self._cache_provider = cache_provider
        self._execution_manager = execution_manager
        self._output_reader = ToolOutputReader(cache_provider, execution_manager)
        self._tools = (
            create_history_tool(self),
            create_search_tool(self),
            create_read_tool(self),
        )

    @property
    def tools(self) -> tuple[Tool, ...]:
        return self._tools

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(tool.name for tool in self._tools)

    def get_execution_history(
        self,
        start_position: int | None = None,
        limit: int = RUNTIME.memory_default_page_limit,
    ) -> PreformattedToolResult:
        """Return overview or one position-based page of node execution history."""
        view = self._history.select(start_position=start_position, limit=limit)
        cache = self._cache_provider()
        root = ET.Element("execution-history")
        append_text_element(root, "total", str(view.total))
        append_text_element(root, "view", view.mode)
        if view.start_position is not None:
            append_text_element(root, "start-position", str(view.start_position))
        if view.end_position is not None:
            append_text_element(root, "end-position", str(view.end_position))
        if view.previous_position is not None:
            append_text_element(root, "previous-position", str(view.previous_position))
        if view.next_position is not None:
            append_text_element(root, "next-position", str(view.next_position))

        for entry in view.entries:
            if (
                view.omitted is not None
                and entry.position == view.omitted.end_position + 1
            ):
                self._append_omitted(root, view.omitted)
            self._append_history_entry(root, entry, cache)
        return PreformattedToolResult(render_xml(root))

    @staticmethod
    def _append_history_entry(
        root: ET.Element,
        entry: ExecutionHistoryEntry,
        cache: NodeCache,
    ) -> None:
        call = ET.SubElement(root, "call")
        append_text_element(call, "position", str(entry.position))
        append_text_element(call, "name", entry.name)
        append_text_element(call, "id", entry.call_id)
        output_id = entry.recoverable_output_id(cache)
        if output_id is not None:
            append_text_element(call, "saved-output-id", output_id)

    @staticmethod
    def _append_omitted(root: ET.Element, omitted: OmittedHistoryRange) -> None:
        element = ET.SubElement(root, "omitted")
        append_text_element(element, "from-position", str(omitted.start_position))
        append_text_element(element, "to-position", str(omitted.end_position))
        append_text_element(element, "count", str(omitted.count))

    def search_tool_outputs(
        self,
        query: str = "",
        tool_name: str | None = None,
        skill_path: str | None = None,
        offset: int = 0,
        limit: int = RUNTIME.memory_default_page_limit,
    ) -> PreformattedToolResult:
        """Search saved and runtime outputs by text, tool name, and skill path."""
        records, total, next_offset = ToolCallSearch(
            self._cache_provider(), self._execution_manager
        ).search_records(
            query=query,
            tool_name=tool_name,
            skill_path=skill_path,
            offset=offset,
            limit=limit,
        )
        root = ET.Element("tool-outputs")
        append_text_element(root, "total", str(total))
        if next_offset is not None:
            append_text_element(root, "next-offset", str(next_offset))
        records_element = ET.SubElement(root, "records")
        for record in records:
            self._append_search_record(records_element, record)
        return PreformattedToolResult(render_xml(root))

    @classmethod
    def _append_search_record(cls, root: ET.Element, record: ToolCallRecord) -> None:
        element = ET.SubElement(root, "record")
        output = ToolOutputView(record)
        for name, value in output.search_fields().items():
            if value is not None:
                append_text_element(element, name, str(value))
        append_text_element(element, "preview", output.preview(cls.PREVIEW_LIMIT))

    def read_tool_output(
        self, output_id: str, cursor: int | None = None
    ) -> PreformattedToolResult:
        """Read current or saved output by its canonical output ID."""
        return self._output_reader.read(output_id, cursor)
