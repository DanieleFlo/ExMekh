from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Final

from qs.core.agents.capabilities.capabilities import Capabilities
from qs.core.agents.capabilities.data import CapabilitiesData
from qs.core.agents.deep_agent.agent_response import AgentResponse
from qs.interfaces.provider import Provider

from .runtime import NodeRuntime, NodeRuntimeConfig
from .types import NodeUserMessage


class Node:
    """Configure one node in a deep-agent hierarchy."""

    REQUIRED_TOOL_CHOICE: Final[str] = NodeRuntime.REQUIRED_TOOL_CHOICE
    EMPTY_RESPONSE_MESSAGE: Final[str] = NodeRuntime.EMPTY_RESPONSE_MESSAGE

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
        self._runtime: NodeRuntime | None = None

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

    def initialize(self, provider: Provider) -> CapabilitiesData:
        if not isinstance(provider, Provider):
            raise TypeError("provider must be a Provider")

        capabilities = self.load_capabilities()
        hds = next(hds for hds in capabilities.hds if hds.name == self._name_hds)
        exit_skill = next(
            skill
            for skill in capabilities.exit_skills
            if skill.name == hds.metadata.exit_skill
        )
        self._runtime = NodeRuntime(
            NodeRuntimeConfig(
                provider=provider,
                capabilities=capabilities,
                hds_name=self._name_hds,
                exit_skill=exit_skill,
                persistent_history=self._persistent_history,
                initial_depth=self._initial_depth,
            )
        )
        return capabilities

    def load_capabilities(self) -> CapabilitiesData:
        capabilities = self._capability_loader.load()
        self._require_configured_hds(capabilities)
        return capabilities

    def run(self, user_message: NodeUserMessage) -> AgentResponse:
        if self._runtime is None:
            raise RuntimeError("Node must be initialized before run")
        return self._runtime.run(user_message)

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
