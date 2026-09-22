from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from xml.etree import ElementTree as ET

from qs.core.cache.node_cache import NodeCache
from qs.core.cache.tool_call_record import ToolCallRecord
from qs.core.models.tool import PreformattedToolResult, Tool
from qs.core.models.xml_header import XmlHeader
from qs.utils.json import stringify_tool_result

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

    def is_recoverable(self, cache: NodeCache) -> bool:
        return (
            self.saved_output_id is not None
            and self.saved_session_dir == cache.session_dir
            and cache.has(self.saved_output_id)
        )


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

    @property
    def entries(self) -> tuple[ExecutionHistoryEntry, ...]:
        return tuple(self._entries)

    def select(
        self,
        *,
        start_position: int | None = None,
        limit: int = 20,
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

        head_count = max(1, limit // 4)
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
        if type(limit) is not int or not 2 <= limit <= 100:
            raise ValueError("limit must be between 2 and 100")


class HarnessMemoryTools:
    """Node-scoped XML tools for inspecting execution and cached outputs."""

    HISTORY_TOOL_NAME = "get_execution_history"
    SEARCH_TOOL_NAME = "search_saved_tool_outputs"
    READ_TOOL_NAME = "read_saved_tool_output"
    PREVIEW_LIMIT = 240

    def __init__(
        self,
        history: ExecutionHistory,
        cache_provider: Callable[[], NodeCache],
    ) -> None:
        self._history = history
        self._cache_provider = cache_provider
        self._tools = (
            Tool(
                self.get_execution_history,
                name=self.HISTORY_TOOL_NAME,
                description=(
                    "Return navigable XML history of capability calls. Omit "
                    "start_position for a head-and-tail overview, or use a "
                    "1-based position and limit to read a page."
                ),
                cache_result=False,
            ),
            Tool(
                self.search_saved_tool_outputs,
                name=self.SEARCH_TOOL_NAME,
                description=(
                    "Search node-scoped saved outputs and return paginated XML "
                    "previews. All filters are optional."
                ),
                cache_result=False,
            ),
            Tool(
                self.read_saved_tool_output,
                name=self.READ_TOOL_NAME,
                description=(
                    "Read one saved output by ID as an XML header followed by "
                    "its complete content."
                ),
                cache_result=False,
            ),
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
        limit: int = 20,
    ) -> PreformattedToolResult:
        """Return overview or one position-based page of node execution history."""
        cache = self._cache_provider()
        view = self._history.select(start_position=start_position, limit=limit)
        root = ET.Element("execution-history")
        ET.SubElement(root, "total").text = str(view.total)
        ET.SubElement(root, "view").text = view.mode
        if view.start_position is not None:
            ET.SubElement(root, "start-position").text = str(view.start_position)
        if view.end_position is not None:
            ET.SubElement(root, "end-position").text = str(view.end_position)
        if view.previous_position is not None:
            ET.SubElement(root, "previous-position").text = str(
                view.previous_position
            )
        if view.next_position is not None:
            ET.SubElement(root, "next-position").text = str(view.next_position)

        for entry in view.entries:
            if (
                view.omitted is not None
                and entry.position == view.omitted.end_position + 1
            ):
                self._append_omitted(root, view.omitted)
            self._append_history_entry(root, entry, cache)
        return PreformattedToolResult(_render_xml(root))

    @staticmethod
    def _append_history_entry(
        root: ET.Element,
        entry: ExecutionHistoryEntry,
        cache: NodeCache,
    ) -> None:
        call = ET.SubElement(root, "call")
        ET.SubElement(call, "position").text = str(entry.position)
        ET.SubElement(call, "name").text = entry.name
        ET.SubElement(call, "id").text = entry.call_id
        if entry.is_recoverable(cache):
            ET.SubElement(call, "saved-output-id").text = entry.saved_output_id

    @staticmethod
    def _append_omitted(root: ET.Element, omitted: OmittedHistoryRange) -> None:
        element = ET.SubElement(root, "omitted")
        ET.SubElement(element, "from-position").text = str(
            omitted.start_position
        )
        ET.SubElement(element, "to-position").text = str(omitted.end_position)
        ET.SubElement(element, "count").text = str(omitted.count)

    def search_saved_tool_outputs(
        self,
        query: str = "",
        tool_name: str | None = None,
        skill_path: str | None = None,
        offset: int = 0,
        limit: int = 20,
    ) -> PreformattedToolResult:
        """Search saved outputs by text, exact tool name, and exact skill path."""
        records, total, next_offset = ToolCallSearch(
            self._cache_provider()
        ).search_records(
            query=query,
            tool_name=tool_name,
            skill_path=skill_path,
            offset=offset,
            limit=limit,
        )
        root = ET.Element("saved-tool-outputs")
        ET.SubElement(root, "total").text = str(total)
        if next_offset is not None:
            ET.SubElement(root, "next-offset").text = str(next_offset)
        records_element = ET.SubElement(root, "records")
        for record in records:
            record_element = ET.SubElement(records_element, "record")
            ET.SubElement(record_element, "name").text = record.name
            ET.SubElement(record_element, "type").text = record.source
            ET.SubElement(record_element, "id").text = record.id
            ET.SubElement(record_element, "skill-path").text = record.skill_path
            ET.SubElement(record_element, "preview").text = self._preview(record)
        return PreformattedToolResult(_render_xml(root))

    def read_saved_tool_output(self, output_id: str) -> PreformattedToolResult:
        """Read one complete saved output by its opaque output ID."""
        record = self._cache_provider().get(output_id)
        if record is None:
            root = ET.Element("saved-tool-output")
            ET.SubElement(root, "id").text = output_id
            ET.SubElement(root, "found").text = "false"
            return PreformattedToolResult(_render_xml(root))

        header = XmlHeader(
            source=record.source,
            name=record.name,
            id=record.id,
            args=record.args,
        )
        return PreformattedToolResult(header.prepend_to(self._content(record)))

    @classmethod
    def _preview(cls, record: ToolCallRecord) -> str:
        preview = " ".join(cls._content(record).split())
        if len(preview) <= cls.PREVIEW_LIMIT:
            return preview
        return preview[: cls.PREVIEW_LIMIT - 1] + "…"

    @staticmethod
    def _content(record: ToolCallRecord) -> str:
        if (
            record.source == "DeepSkill"
            and isinstance(record.result, dict)
            and isinstance(record.result.get("text"), str)
        ):
            return record.result["text"]
        return stringify_tool_result(record.result)


def _render_xml(root: ET.Element) -> str:
    ET.indent(root)
    return ET.tostring(root, encoding="unicode", short_empty_elements=True)
