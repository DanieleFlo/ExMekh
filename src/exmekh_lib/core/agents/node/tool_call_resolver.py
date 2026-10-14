from exmekh_lib.config.runtime_text import RUNTIME_TEXT
from exmekh_lib.core.agents.capabilities.data import CapabilitiesData
from exmekh_lib.core.agents.deep_skill.deep_skill import CalledDeepSkill, DeepSkill
from exmekh_lib.core.agents.exit_tool.exit_tool import CalledExitTool, ExitTool
from exmekh_lib.core.models.assistant_message import ToolCall
from exmekh_lib.core.models.tool import CalledTool, Tool

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
        registered_tools = (*self._capabilities.get_all_tools(), *self._runtime_tools)
        for tool in registered_tools:
            if not isinstance(tool, Tool):
                raise TypeError(
                    RUNTIME_TEXT.tool_is_not_a_valid_deep_skill_exit_tool_2.format(
                        tool_name=tool_name
                    )
                )
            if tool.get_name() != tool_name:
                continue
            if isinstance(tool, DeepSkill):
                return CalledDeepSkill(deep_skill=tool, tool_call=tool_call)
            if isinstance(tool, ExitTool):
                return CalledExitTool(exit_tool=tool, tool_call=tool_call)
            return CalledTool(tool=tool, tool_call=tool_call)
        raise ValueError(
            RUNTIME_TEXT.tool_is_not_a_valid_deep_skill_exit_tool.format(
                tool_name=tool_name,
                available=", ".join(tool.get_name() for tool in registered_tools),
            )
        )
