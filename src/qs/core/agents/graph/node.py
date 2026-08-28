from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

from qs.core.agents.capabilities.capabilities import Capabilities
from qs.core.agents.capabilities.data import CapabilitiesData
from qs.core.agents.deep_agent.agent_response import AgentResponse
from qs.core.agents.deep_skill.deep_skill import CalledDeepSkill, DeepSkill
from qs.core.agents.deep_skill.execution import DeepSkillExecutionTracker
from qs.core.agents.exit_skill.exit_skill import CalledExitSkill, ExitSkill
from qs.core.agents.exit_tool.exit_tool import CalledExitTool, ExitTool
from qs.core.models.assistant_message import ToolCall
from qs.core.models.client import Client
from qs.core.models.client_request import ClientRequest
from qs.core.models.provider_response import ProviderResponse
from qs.core.models.system_message import SystemMessage
from qs.core.models.tool import CalledTool, Tool
from qs.core.models.tool_message import ToolMessage
from qs.core.models.tools import Tools
from qs.core.models.user_message import (
    FilesUserMessage,
    ImagesUserMessage,
    TextsUserMessage,
    UserMessage,
)
from qs.interfaces.provider import Provider

type NodeUserMessage = str | TextsUserMessage | ImagesUserMessage | FilesUserMessage

logger = logging.getLogger(__name__)


class Node:
    """Configuration and conversational runtime of one agent node."""

    REQUIRED_TOOL_CHOICE: Final[str] = "required"

    def __init__(
        self,
        capabilities_paths: Sequence[Path],
        name_hds: str,
        children: Sequence[Node] = (),
        persistent_history: bool = True,
        depth: int = 0,
    ) -> None:
        self._capability_loader = Capabilities(capabilities_paths)
        self._name_hds = self._validate_hds_name(name_hds)
        self._children = self._validate_children(children)
        self._persistent_history = self._validate_persistent_history(persistent_history)
        self._initial_depth = self._validate_depth(depth)
        self._depth = self._initial_depth
        self._skill_executions = DeepSkillExecutionTracker()

        self._provider: Provider | None = None
        self._loaded_capabilities: CapabilitiesData | None = None
        self._client: Client | None = None
        self._exit_skill: ExitSkill | None = None
        self._exit_tool: ExitTool | None = None

    @property
    def name_hds(self) -> str:
        return self._name_hds

    @property
    def children(self) -> tuple[Node, ...]:
        return self._children

    @property
    def persistent_history(self) -> bool:
        return self._persistent_history

    @property
    def depth(self) -> int:
        return self._depth

    def initialize(self, provider: Provider) -> CapabilitiesData:
        """Load, validate and bind this node to a provider."""
        if not isinstance(provider, Provider):
            raise TypeError("provider must be a Provider")

        capabilities = self.load_capabilities()
        hds = next(hds for hds in capabilities.hds if hds.name == self._name_hds)
        exit_skill = next(
            skill
            for skill in capabilities.exit_skills
            if skill.name == hds.metadata.exit_skill
        )
        exit_tool = next(
            tool
            for tool in capabilities.exit_tools
            if tool.exit_skill_name == exit_skill.name
        )

        self._provider = provider
        self._loaded_capabilities = capabilities
        self._exit_skill = exit_skill
        self._exit_tool = exit_tool
        self._depth = self._initial_depth
        self._skill_executions = DeepSkillExecutionTracker()
        self._client = self._build_client()
        return capabilities

    def load_capabilities(self) -> CapabilitiesData:
        """Load this node's capabilities and verify its configured HDS."""
        capabilities = self._capability_loader.load()
        self._require_configured_hds(capabilities)
        return capabilities

    def run(self, user_message: NodeUserMessage) -> AgentResponse:
        client = self._client_for_run()

        logger.info("Starting node run: node=%s depth=%d", self._name_hds, self._depth)
        response = client.call(
            ClientRequest(
                message=UserMessage(content=user_message),
                tool_choice=self._current_tool_choice(),
            )
        )
        response = self._routing_loop(response)
        logger.info(
            "Completed node run: node=%s depth=%d root_executions=%d",
            self._name_hds,
            self._depth,
            len(self._skill_executions.roots),
        )

        return AgentResponse(
            response=response,
            usage=response.usage,
            thinking=response.thinking,
            history=self._client.dump_history(),
            skill_executions=self._skill_executions.roots,
        )

    def _routing_loop(self, response: ProviderResponse) -> ProviderResponse:
        if self._client is None:
            raise RuntimeError("Node must be initialized before routing loop")
        while self._has_tool_calls(response.get_tool_calls()):
            tool_calls = response.get_tool_calls()
            logger.info(
                "Routing assistant tool calls: count=%d depth=%d",
                len(tool_calls or ()),
                self._depth,
            )
            list_of_tool_calls = self._map_tool_call_to_object(tool_calls)
            for capability in list_of_tool_calls:
                if isinstance(capability, CalledDeepSkill):
                    self._run_deep_skill(capability)
                elif isinstance(capability, CalledTool):
                    self._run_tool(capability)
                elif isinstance(capability, CalledExitSkill):
                    self._run_exit_skill(capability)
                elif isinstance(capability, CalledExitTool):
                    self._run_exit_tool(capability)
                else:
                    raise TypeError(
                        "Node supports only deep skills, exit skills, exit tools, and tools"
                    )
            response = self._client.continue_after_tools(
                tool_choice=self._current_tool_choice()
            )
        message = response.get_message()
        if not isinstance(message, str):
            raise TypeError("provider response message must be a string")
        return response

    def _map_tool_call_to_object(
        self, tool_calls: list[ToolCall] | None
    ) -> list[CalledDeepSkill | CalledTool | CalledExitSkill | CalledExitTool]:
        if tool_calls is None:
            return []
        return [self._identify_tool_choice(tool_call) for tool_call in tool_calls or ()]

    def _identify_tool_choice(
        self, tool_call: ToolCall
    ) -> CalledDeepSkill | CalledTool | CalledExitSkill | CalledExitTool:
        if self._loaded_capabilities is None or self._exit_tool is None:
            raise RuntimeError(
                "Node must be initialized before identifying tool choice"
            )

        name_tool = tool_call.get_name()
        logger.info(
            "Identifying tool call: name=%s call_id=%s depth=%d",
            name_tool,
            tool_call.id,
            self._depth,
        )
        capabilities = self._loaded_capabilities.get_all_tools()
        for tool in capabilities:
            if tool.name == name_tool:
                # ExitSkill is a instance of DeepSkill, so we need to check for ExitSkill first
                if isinstance(tool, ExitSkill):
                    return CalledExitSkill(exit_skill=tool, tool_call=tool_call)
                elif isinstance(tool, DeepSkill):
                    return CalledDeepSkill(deep_skill=tool, tool_call=tool_call)
                elif isinstance(tool, ExitTool):
                    return CalledExitTool(exit_tool=tool, tool_call=tool_call)
                elif isinstance(tool, Tool):
                    return CalledTool(tool=tool, tool_call=tool_call)
                else:
                    raise TypeError(
                        f"Tool '{name_tool}' is not a valid deep skill, exit skill, exit tool, or tool instance"
                    )
        raise ValueError(
            f"Tool '{name_tool}' is not a valid deep skill, exit skill, exit tool, or tool"
        )

    def _run_deep_skill(self, called_deep_skill: CalledDeepSkill) -> None:
        if self._client is None:
            raise RuntimeError("Node must be initialized before running a deep skill")

        self._open_deep_skill(called_deep_skill)
        content = self._render_deep_skill(called_deep_skill.deep_skill)
        self._client.add_tool_response(
            ToolMessage(
                content=content,
                tool_call_id=called_deep_skill.tool_call.id,
                name=called_deep_skill.deep_skill.name,
            )
        )

    def _run_tool(self, called_tool: CalledTool) -> None:
        if self._client is None:
            raise RuntimeError("Node must be initialized before running a tool")
        logger.info(
            "Executing tool: name=%s call_id=%s depth=%d",
            called_tool.name,
            called_tool.call_id,
            self._depth,
        )
        result_tool = called_tool.execute()
        content = self._stringify_result(result_tool)
        self._client.add_tool_response(
            ToolMessage(
                content=content,
                tool_call_id=called_tool.call_id,
                name=called_tool.name,
            )
        )

    def _run_exit_skill(self, called_exit_skill: CalledExitSkill) -> None:
        if self._client is None:
            raise RuntimeError("Node must be initialized before running an exit skill")

        logger.info(
            "Rendering exit skill: name=%s call_id=%s depth=%d",
            called_exit_skill.exit_skill.name,
            called_exit_skill.tool_call.id,
            self._depth,
        )
        content = called_exit_skill.exit_skill.get_body()
        self._client.add_tool_response(
            ToolMessage(
                content=content,
                tool_call_id=called_exit_skill.tool_call.id,
                name=called_exit_skill.exit_skill.name,
            )
        )

    def _run_exit_tool(self, called_exit_tool: CalledExitTool) -> None:
        if self._client is None:
            raise RuntimeError("Node must be initialized before running an exit tool")
        arguments = called_exit_tool.tool_call.get_arguments()
        logger.info(
            "Executing exit tool: name=%s call_id=%s active_skills=%d depth=%d",
            called_exit_tool.exit_tool.name,
            called_exit_tool.call_id,
            self._skill_executions.depth,
            self._depth,
        )
        result_tool = called_exit_tool.exit_tool.execute(**arguments)
        content = self._get_exit_output(result_tool)
        if self._skill_executions.depth:
            # Close the current execution and use the exit result as the response
            # to the tool call that opened the deep skill.
            self._close_deep_skill(content)
            return

        self._depth = max(0, self._depth - 1)
        logger.warning(
            "Exit tool executed without an active deep skill: call_id=%s depth=%d",
            called_exit_tool.call_id,
            self._depth,
        )
        self._client.add_tool_response(
            ToolMessage(
                content=content,
                tool_call_id=called_exit_tool.call_id,
                name=called_exit_tool.exit_tool.name,
            )
        )

    def _client_for_run(self) -> Client:
        if self._client is None:
            raise RuntimeError("Node must be initialized before run")

        if not self._persistent_history:
            self._depth = self._initial_depth
            self._skill_executions = DeepSkillExecutionTracker()
            self._client = self._build_client()

        return self._client

    def _build_client(self) -> Client:
        if (  # pragma: no cover - guarded by initialize
            self._provider is None
            or self._loaded_capabilities is None
            or self._exit_skill is None
            or self._exit_tool is None
        ):
            raise RuntimeError("Node is not initialized")

        hds = next(
            hds for hds in self._loaded_capabilities.hds if hds.name == self._name_hds
        )
        tools = Tools(list(self._loaded_capabilities.get_all_tools()))
        return Client(
            provider=self._provider,
            system_message=SystemMessage(content=hds.get_hds()),
            tools=tools,
        )

    def _render_deep_skill(self, deep_skill: DeepSkill) -> str:
        if self._exit_skill is None:  # pragma: no cover - guarded by initialize
            raise RuntimeError("Node has no exit skill")

        return deep_skill.get_skill(self._exit_skill)

    def _open_deep_skill(self, called_deep_skill: CalledDeepSkill) -> None:
        if self._client is None:  # pragma: no cover - guarded by caller
            raise RuntimeError("Node must be initialized before opening a deep skill")

        start_index = len(self._client.get_history()) - 1
        self._skill_executions.open(
            called_deep_skill.deep_skill,
            called_deep_skill.tool_call.id,
            start_index,
        )
        self._depth += 1
        logger.info(
            "Opened deep skill: name=%s call_id=%s start_index=%d depth=%d",
            called_deep_skill.deep_skill.name,
            called_deep_skill.tool_call.id,
            start_index,
            self._depth,
        )

    def _close_deep_skill(self, output: str) -> None:
        if self._client is None:  # pragma: no cover - guarded by caller
            raise RuntimeError("Node must be initialized before closing a deep skill")

        history = self._client.copy_history()
        start_index = self._skill_executions.current.start_index
        execution = self._skill_executions.close(
            end_index=len(history) - 1,
            output=output,
            history=history[start_index:],
        )
        response = ToolMessage(
            content=output,
            tool_call_id=execution.opening_tool_call_id,
            name=execution.skill.name,
        )
        self._client.replace_history_from(execution.start_index + 1, [response])
        self._depth = max(0, self._depth - 1)
        logger.info(
            "Closed deep skill: name=%s opening_call_id=%s start_index=%d "
            "end_index=%d transcript_messages=%d compacted_history=%d depth=%d",
            execution.skill.name,
            execution.opening_tool_call_id,
            execution.start_index,
            execution.end_index,
            len(execution.history),
            len(self._client.get_history()),
            self._depth,
        )

    @staticmethod
    def _get_exit_output(result: Any) -> str:
        if not isinstance(result, dict) or not isinstance(result.get("text"), str):
            raise TypeError("exit tool result must contain a text string")
        return result["text"]

    def _current_tool_choice(self) -> str | None:
        if self._depth > 0:
            return self.REQUIRED_TOOL_CHOICE

        return None

    @staticmethod
    def _has_tool_calls(tool_calls: list[ToolCall] | None) -> bool:
        if tool_calls is None:
            return False
        if not isinstance(tool_calls, list):
            raise TypeError("tool_calls must be a list or None")
        return bool(tool_calls)

    @staticmethod
    def _stringify_result(result: Any) -> str:
        if isinstance(result, str):
            return result

        return json.dumps(result, ensure_ascii=True, default=str)

    @staticmethod
    def _validate_hds_name(name_hds: str) -> str:
        if not isinstance(name_hds, str):
            raise TypeError("name_hds must be a string")

        if not name_hds.strip():
            raise ValueError("name_hds cannot be empty")

        return name_hds

    @staticmethod
    def _validate_children(children: Sequence[Node]) -> tuple[Node, ...]:
        if isinstance(children, (str, bytes)) or not isinstance(children, Sequence):
            raise TypeError("children must be a sequence of Node")

        nodes = tuple(children)
        if not all(isinstance(child, Node) for child in nodes):
            raise TypeError("children must contain only Node instances")

        return nodes

    @staticmethod
    def _validate_persistent_history(persistent_history: bool) -> bool:
        if not isinstance(persistent_history, bool):
            raise TypeError("persistent_history must be a bool")

        return persistent_history

    @staticmethod
    def _validate_depth(depth: int) -> int:
        if not isinstance(depth, int) or isinstance(depth, bool):
            raise TypeError("depth must be an integer")

        if depth < 0:
            raise ValueError("depth cannot be negative")

        return depth

    def _require_configured_hds(self, capabilities: CapabilitiesData) -> None:
        available_hds_names = tuple(hds.name for hds in capabilities.hds)
        if self._name_hds in available_hds_names:
            return

        available_hds = ", ".join(available_hds_names) or "none"
        raise ValueError(
            f"HDS '{self._name_hds}' not found. Available HDS: {available_hds}"
        )
