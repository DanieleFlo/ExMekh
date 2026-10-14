from __future__ import annotations

from pathlib import Path
from typing import Any

from exmekh_lib.config.runtime import RUNTIME
from exmekh_lib.core.agents.tools.deep_tool import DeepTool
from exmekh_lib.core.cache.node_cache import NodeCache
from exmekh_lib.core.cache.tool_call_record import ToolCallRecord
from exmekh_lib.core.models.tool import Tool
from exmekh_lib.core.tool_calls.execution_manager import ToolExecutionManager
from exmekh_lib.core.tool_calls.output_preparer import ToolOutputPreparer
from exmekh_lib.core.tool_calls.output_runtime import ToolOutputRuntime
from exmekh_lib.core.tool_calls.output_view import ToolOutputView

from .query import ToolOutputQuery
from .raw_search import run as run_raw_search


class ToolCallSearch:
    """Search one node's stored and runtime outputs without consuming results."""

    def __init__(
        self,
        cache: NodeCache,
        execution_manager: ToolExecutionManager | None = None,
        output_preparer: ToolOutputPreparer | None = None,
    ) -> None:
        self._cache = cache
        self._output_runtime = ToolOutputRuntime(execution_manager, output_preparer)

    def as_tool(self) -> Tool:
        return DeepTool(run_raw_search, Path.cwd(), self.search, cache_result=False)

    def search(
        self,
        query: str = "",
        tool_name: str | None = None,
        skill_path: str | None = None,
        offset: int = 0,
        limit: int = RUNTIME.memory_default_page_limit,
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
        limit: int = RUNTIME.memory_default_page_limit,
    ) -> tuple[list[ToolCallRecord], int, int | None]:
        """Return matching records and pagination metadata."""
        return self.find(
            ToolOutputQuery(
                query=query,
                tool_name=tool_name,
                skill_path=skill_path,
                offset=offset,
                limit=limit,
            )
        )

    def find(
        self, query: ToolOutputQuery
    ) -> tuple[list[ToolCallRecord], int, int | None]:
        return query.select(self._output_records())

    def _output_records(self) -> tuple[ToolCallRecord, ...]:
        records = {
            record.id: ToolOutputView(record).saved_record()
            for record in self._cache.get_all()
        }
        return self._output_runtime.prepare_records(records)
