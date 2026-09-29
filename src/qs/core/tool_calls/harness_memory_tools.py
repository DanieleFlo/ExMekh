from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from qs.capabilities.tools.get_execution_history.rendering import (
    render_execution_history,
)
from qs.capabilities.tools.get_execution_history.tool import (
    create_tool as create_history_tool,
)
from qs.capabilities.tools.read_tool_output.tool import (
    create_tool as create_read_tool,
)
from qs.capabilities.tools.read_tool_output.tool_output_reader import ToolOutputReader
from qs.capabilities.tools.search_tool_outputs.rendering import render_search_results
from qs.capabilities.tools.search_tool_outputs.tool import (
    create_tool as create_search_tool,
)
from qs.config.runtime import RUNTIME
from qs.core.cache.node_cache import NodeCache
from qs.core.models.tool import PreformattedToolResult, Tool

from .execution_history import ExecutionHistory
from .execution_manager import ToolExecutionManager


class HarnessMemoryTools:
    """Node-scoped XML tools for inspecting execution and cached outputs."""

    HISTORY_TOOL_NAME = RUNTIME.history_tool_name
    SEARCH_TOOL_NAME = RUNTIME.search_tool_name
    READ_TOOL_NAME = RUNTIME.read_tool_name

    def __init__(
        self,
        history: ExecutionHistory,
        cache_provider: Callable[[], NodeCache],
        execution_manager: ToolExecutionManager | None = None,
        work_root: Path | None = None,
    ) -> None:
        resolved_work_root = work_root or Path.cwd()
        self._history = history
        self._cache_provider = cache_provider
        self._execution_manager = execution_manager
        self._output_reader = ToolOutputReader(cache_provider, execution_manager)
        self._tools = (
            create_history_tool(self.get_execution_history, resolved_work_root),
            create_search_tool(self.search_tool_outputs, resolved_work_root),
            create_read_tool(self.read_tool_output, resolved_work_root),
        )

    @property
    def tools(self) -> tuple[Tool, ...]:
        return self._tools

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(tool.get_name() for tool in self._tools)

    def get_execution_history(
        self,
        start_position: int | None = None,
        limit: int = RUNTIME.memory_default_page_limit,
    ) -> PreformattedToolResult:
        return render_execution_history(
            self._history, self._cache_provider(), start_position, limit
        )

    def search_tool_outputs(
        self,
        query: str = "",
        tool_name: str | None = None,
        skill_path: str | None = None,
        offset: int = 0,
        limit: int = RUNTIME.memory_default_page_limit,
    ) -> PreformattedToolResult:
        return render_search_results(
            self._cache_provider(),
            self._execution_manager,
            query,
            tool_name,
            skill_path,
            offset,
            limit,
        )

    def read_tool_output(
        self, output_id: str, cursor: int | None = None
    ) -> PreformattedToolResult:
        return self._output_reader.read(output_id, cursor)
