from __future__ import annotations

import json
from typing import Any

from qs.config.runtime import RUNTIME
from qs.core.cache.node_cache import NodeCache
from qs.core.cache.tool_call_record import ToolCallRecord
from qs.core.models.tool import Tool

from .execution_manager import ExecutionSnapshot, ToolExecutionManager
from .output_view import ToolOutputView


class ToolCallSearch:
    """Search one node's stored and runtime outputs without consuming results."""

    def __init__(
        self, cache: NodeCache, execution_manager: ToolExecutionManager | None = None
    ) -> None:
        self._cache = cache
        self._execution_manager = execution_manager

    def as_tool(self) -> Tool:
        return Tool(
            self.search,
            name=RUNTIME.search_tool_name,
            cache_result=False,
            background_enabled=False,
        )

    def search(
        self,
        query: str = "",
        tool_name: str | None = None,
        skill_path: str | None = None,
        offset: int = 0,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Search saved and runtime calls by text and exact skill path, with pagination.

        Searches names, IDs, arguments and outputs. Returns only this node's
        records. Omit skill_path to search all its skill paths.
        """
        records, total, next_offset = self.search_records(
            query=query,
            tool_name=tool_name,
            skill_path=skill_path,
            offset=offset,
            limit=limit,
        )
        return {
            "records": [record.to_dict() for record in records],
            "total": total,
            "next_offset": next_offset,
        }

    def search_records(
        self,
        query: str = "",
        tool_name: str | None = None,
        skill_path: str | None = None,
        offset: int = 0,
        limit: int = 20,
    ) -> tuple[list[ToolCallRecord], int, int | None]:
        """Return matching records and pagination metadata."""
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if tool_name is not None and (
            not isinstance(tool_name, str) or not tool_name.strip()
        ):
            raise ValueError("tool_name must be a non-empty string or None")
        if skill_path is not None and (
            not isinstance(skill_path, str) or not skill_path.startswith("/")
        ):
            raise ValueError("skill_path must be an absolute skill path")
        if type(offset) is not int or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        needle = query.casefold()
        matches = [
            record
            for record in self._output_records()
            if (tool_name is None or record.name == tool_name)
            and (skill_path is None or record.skill_path == skill_path)
            and needle in json.dumps(record.to_dict(), ensure_ascii=False).casefold()
        ]
        end = offset + limit
        return matches[offset:end], len(matches), end if end < len(matches) else None

    def _output_records(self) -> tuple[ToolCallRecord, ...]:
        records = {
            record.id: ToolOutputView(record).saved_record()
            for record in self._cache.get_all()
        }
        if self._execution_manager is not None:
            for snapshot in self._execution_manager.output_snapshots():
                record = self._execution_record(
                    snapshot, records.get(snapshot.output_id)
                )
                records[record.id] = record
        return tuple(records.values())

    def _execution_record(
        self, snapshot: ExecutionSnapshot, stored: ToolCallRecord | None
    ) -> ToolCallRecord:
        if snapshot.output_id is None:
            raise ValueError("A searchable execution must have an output ID")
        result = snapshot.record_result()
        execution = {
            **snapshot.status_info(),
            "response_type": snapshot.response_type,
            "storage": "saved" if stored is not None else "runtime",
        }
        if not snapshot.is_finished:
            try:
                partial = self._execution_manager.partial_output(snapshot.output_id, 0)
                if partial is not None and partial[0]:
                    result = {**result, "partial_output": partial[0]}
                    execution["response_type"] = "partial"
            except Exception as error:
                result = {**result, "partial_output_error": str(error)}
        return ToolCallRecord(
            name=snapshot.name,
            id=snapshot.output_id,
            args=snapshot.arguments,
            result=result,
            skill_path=snapshot.skill_path,
            tool_call_id=snapshot.call_id,
            thinking=stored.thinking if stored is not None else None,
            execution=execution,
        )
