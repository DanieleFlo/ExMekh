from __future__ import annotations

from collections.abc import Sequence

from qs.core.agents.capabilities.defaults import DefaultCapabilities
from qs.core.models.tool import Tool


class _DefaultToolComposer:
    """Apply one node's hook and runtime override configuration to tools."""

    def __init__(self, defaults: DefaultCapabilities | None) -> None:
        self._defaults = defaults

    def apply_hooks[ToolType: Tool](self, tool: ToolType) -> ToolType:
        if self._defaults is None:
            return tool
        hooks = self._defaults.tool_hooks_for(tool.get_name())
        return tool if hooks is None else tool.with_hooks(hooks)

    def compose_runtime_tool(self, tool: Tool) -> Tool | None:
        if self._defaults is None:
            return tool
        name = tool.get_name()
        if not self._defaults.includes(name):
            return None
        replacement = self._defaults.tool_override(name)
        composed = self.apply_hooks(tool if replacement is None else replacement)
        if not tool.background_enabled:
            composed = composed.without_background_execution()
        return composed

    def compose_runtime_tools(self, tools: Sequence[Tool]) -> tuple[Tool, ...]:
        composed = []
        for tool in tools:
            configured = self.compose_runtime_tool(tool)
            if configured is not None:
                composed.append(configured)
        return tuple(composed)
