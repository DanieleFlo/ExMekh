from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from exmekh_lib.capabilities.tools.get_execution_history.rendering import (
    render_execution_history,
)
from exmekh_lib.capabilities.tools.get_execution_history.tool import (
    create_tool as create_history_tool,
)
from exmekh_lib.capabilities.tools.read_tool_output.summary import (
    SummaryMode,
    ToolOutputSummarizer,
)
from exmekh_lib.capabilities.tools.read_tool_output.tool import (
    create_tool as create_read_tool,
)
from exmekh_lib.capabilities.tools.read_tool_output.tool_output_reader import (
    ToolOutputReader,
)
from exmekh_lib.capabilities.tools.search_tool_outputs.query import ToolOutputQuery
from exmekh_lib.capabilities.tools.search_tool_outputs.rendering import (
    render_search_results,
)
from exmekh_lib.capabilities.tools.search_tool_outputs.tool import (
    create_tool as create_search_tool,
)
from exmekh_lib.capabilities.tools.search_tool_outputs.tool_call_search import (
    ToolCallSearch,
)
from exmekh_lib.config.runtime import RUNTIME
from exmekh_lib.core.cache.node_cache import NodeCache
from exmekh_lib.core.models.tool import PreformattedToolResult, Tool
from exmekh_lib.interfaces.provider import Provider

from .execution_history import ExecutionHistory
from .execution_manager import ToolExecutionManager
from .output_preparer import ToolOutputPreparer


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
        provider: Provider | None = None,
        output_preparer: ToolOutputPreparer | None = None,
    ) -> None:
        resolved_work_root = work_root or Path.cwd()
        self._history = history
        self._cache_provider = cache_provider
        self._execution_manager = execution_manager
        self._output_preparer = output_preparer
        self._output_reader = ToolOutputReader(
            cache_provider,
            execution_manager,
            ToolOutputSummarizer(provider) if provider is not None else None,
            output_preparer,
        )
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
            ToolCallSearch(
                self._cache_provider(), self._execution_manager, self._output_preparer
            ),
            ToolOutputQuery(
                query=query,
                tool_name=tool_name,
                skill_path=skill_path,
                offset=offset,
                limit=limit,
            ),
        )

    def read_tool_output(
        self,
        output_id: str,
        cursor: int | None = None,
        summary_mode: SummaryMode = "none",
    ) -> PreformattedToolResult:
        return self._output_reader.read(output_id, cursor, summary_mode)
