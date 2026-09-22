from qs.core.agents.capabilities.data import CapabilitiesData
from qs.core.agents.deep_skill.deep_skill import CalledDeepSkill, DeepSkill
from qs.core.agents.exit_tool.exit_tool import CalledExitTool, ExitTool
from qs.core.models.assistant_message import ToolCall
from qs.core.models.tool import CalledTool, Tool

type ResolvedToolCall = CalledDeepSkill | CalledTool | CalledExitTool


class ToolCallResolver:
    """Resolve provider tool calls to executable agent capabilities."""

    def __init__(
        self,
        capabilities: CapabilitiesData,
        runtime_tools: tuple[Tool, ...] = (),
    ) -> None:
        self._capabilities = capabilities
        self._runtime_tools = runtime_tools

    def resolve_all(self, tool_calls: list[ToolCall] | None) -> list[ResolvedToolCall]:
        if tool_calls is None:
            return []
        return [self.resolve(tool_call) for tool_call in tool_calls]

    def resolve(self, tool_call: ToolCall) -> ResolvedToolCall:
        tool_name = tool_call.get_name()
        for tool in (*self._capabilities.get_all_tools(), *self._runtime_tools):
            if tool.name != tool_name:
                continue
            if isinstance(tool, DeepSkill):
                return CalledDeepSkill(deep_skill=tool, tool_call=tool_call)
            if isinstance(tool, ExitTool):
                return CalledExitTool(exit_tool=tool, tool_call=tool_call)
            if isinstance(tool, Tool):
                return CalledTool(tool=tool, tool_call=tool_call)
            raise TypeError(
                f"Tool '{tool_name}' is not a valid deep skill, "
                "exit tool, or tool instance"
            )
        raise ValueError(
            f"Tool '{tool_name}' is not a valid deep skill, exit tool, or tool"
        )
