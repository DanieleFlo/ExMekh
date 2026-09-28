from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Final, Literal

from qs.capabilities.tools.call_sub_agent.main import SubAgentTool
from qs.config.runtime import RUNTIME
from qs.core.agents.capabilities.capabilities import Capabilities
from qs.core.agents.capabilities.data import CapabilitiesData
from qs.core.agents.capabilities.defaults import DefaultCapabilities
from qs.core.agents.deep_agent.agent_response import AgentResponse
from qs.core.agents.hds.hds import HDS
from qs.core.cache.node_cache import NodeCache
from qs.core.models.tool import PreformattedToolResult, Tool
from qs.core.tool_calls.memory_tools import ExecutionHistory
from qs.interfaces.event_handler import DeepAgentEventHandler
from qs.interfaces.message import Message
from qs.interfaces.provider import Provider

from .node_runtime import NodeRuntime, NodeRuntimeConfig
from .types import NodeUserMessage


class Node:
    """Configure one node in a deep-agent hierarchy."""

    REQUIRED_TOOL_CHOICE: Final[str] = NodeRuntime.REQUIRED_TOOL_CHOICE
    EMPTY_RESPONSE_MESSAGE: Final[str] = NodeRuntime.EMPTY_RESPONSE_MESSAGE

    def __init__(
        self,
        capabilities_paths: Sequence[Path] | None = None,
        name_hds: str = "",
        children: Sequence[Node] = (),
        persistent_history: bool = True,
        depth: int = 0,
        agent_context_id: str | None = None,
        *,
        default_capabilities: DefaultCapabilities | None = None,
    ) -> None:
        if capabilities_paths is None or (
            isinstance(capabilities_paths, Sequence)
            and not isinstance(capabilities_paths, (str, bytes))
            and not capabilities_paths
        ):
            capabilities_paths = None
        self._capability_loader = Capabilities(
            capabilities_paths, default_capabilities=default_capabilities
        )
        self._default_capabilities = default_capabilities
        self._resolved_default_capabilities = default_capabilities
        self._name_hds = self._validate_hds_name(name_hds)
        self._children = self._validate_children(children)
        self._persistent_history = self._validate_persistent_history(persistent_history)
        self._initial_depth = self._validate_depth(depth)
        self._agent_context_id = self._validate_agent_context_id(agent_context_id)
        self._runtime: NodeRuntime | None = None
        self._hds: HDS | None = None
        self._sub_agent_tool: SubAgentTool | None = None

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
        if self._runtime is None:
            return self._initial_depth
        return self._runtime.depth

    @property
    def agent_context_id(self) -> str | None:
        if self._agent_context_id is not None:
            return self._agent_context_id
        if self._runtime is not None:
            return self._runtime.agent_context_id
        return None

    def initialize(
        self,
        provider: Provider,
        *,
        work_root: Path,
        on_stream: DeepAgentEventHandler | None = None,
    ) -> CapabilitiesData:
        if not isinstance(provider, Provider):
            raise TypeError("provider must be a Provider")
        if on_stream is not None:
            DeepAgentEventHandler.validate_instance(on_stream)

        self._resolved_default_capabilities = self._default_capabilities
        capabilities = self.load_capabilities(work_root=work_root)
        defaults = self._resolved_default_capabilities
        hds = capabilities.get_hds(self._name_hds)
        exit_tool_name = hds.get_exit_tool_name()
        exit_tool = capabilities.get_exit_tool(exit_tool_name)
        runtime_tools = self._build_runtime_tools(defaults)
        context_id = self._agent_context_id or hds.get_agent_context_id()
        self._runtime = NodeRuntime(
            NodeRuntimeConfig(
                provider=provider,
                capabilities=capabilities,
                hds=hds,
                exit_tool=exit_tool,
                persistent_history=self._persistent_history,
                initial_depth=self._initial_depth,
                work_root=work_root,
                agent_context_id=context_id,
                runtime_tools=runtime_tools,
                default_capabilities=defaults,
                on_stream=on_stream,
            )
        )
        self._hds = hds
        return capabilities

    def _build_runtime_tools(
        self, defaults: DefaultCapabilities | None
    ) -> tuple[Tool, ...]:
        self._sub_agent_tool = None
        if not self._children:
            if (
                defaults is not None
                and RUNTIME.sub_agent_tool_name in defaults.overrides
            ):
                raise ValueError("call_sub_agent requires at least one child node")
            return ()
        if defaults is not None and not defaults.includes(RUNTIME.sub_agent_tool_name):
            return ()
        if defaults is not None:
            override = defaults.tool_override(RUNTIME.sub_agent_tool_name)
            if override is not None:
                return (override.without_background_execution(),)
        self._sub_agent_tool = SubAgentTool(self._children)
        return (self._sub_agent_tool,)

    def get_hds(self) -> HDS:
        if self._hds is None:
            raise RuntimeError("Node must be initialized before accessing its HDS")
        return self._hds

    def update_sub_agent_tool_description(self) -> None:
        if self._runtime is None:
            raise RuntimeError("Node must be initialized before updating sub-agents")
        if self._sub_agent_tool is not None:
            child_hds = (child.get_hds() for child in self._children)
            self._sub_agent_tool.set_child_descriptions(
                {hds.get_name(): hds.get_description() for hds in child_hds}
            )

    def load_capabilities(self, *, work_root: Path) -> CapabilitiesData:
        capabilities, defaults = self._capability_loader.load_with_defaults(
            work_root=work_root
        )
        capabilities.get_hds(self._name_hds)
        self._resolved_default_capabilities = defaults
        return capabilities

    def run(self, user_message: NodeUserMessage) -> AgentResponse:
        if self._runtime is None:
            raise RuntimeError("Node must be initialized before run")
        return self._runtime.run(user_message)

    def control_tool_execution(
        self,
        tool_call_id: str,
        action: Literal["wait", "cancel"] = "wait",
        wait_seconds: float = RUNTIME.default_tool_wait_seconds,
    ) -> dict[str, Any]:
        """Control node-local work without changing the conversation history."""
        if self._runtime is None:
            raise RuntimeError("Node must be initialized before controlling tool work")
        return self._runtime.control_tool_execution(tool_call_id, action, wait_seconds)

    def read_tool_output(
        self, output_id: str, cursor: int | None = None
    ) -> PreformattedToolResult:
        """Read node-local output; a final read acknowledges automatic delivery."""
        if self._runtime is None:
            raise RuntimeError("Node must be initialized before reading tool output")
        return self._runtime.read_tool_output(output_id, cursor)

    def restore_history(
        self,
        messages: Sequence[Message[Any]],
        restore_execution_state: Callable[[ExecutionHistory, NodeCache], None]
        | None = None,
    ) -> None:
        """Restore a saved conversation before the next run."""
        if self._runtime is None:
            raise RuntimeError("Node must be initialized before restoring history")
        self._runtime.restore_history(messages, restore_execution_state)

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

    @staticmethod
    def _validate_agent_context_id(agent_context_id: str | None) -> str | None:
        if agent_context_id is None:
            return None
        if not isinstance(agent_context_id, str):
            raise TypeError("agent_context_id must be a string")
        if not agent_context_id.strip():
            raise ValueError("agent_context_id cannot be empty")
        return agent_context_id
