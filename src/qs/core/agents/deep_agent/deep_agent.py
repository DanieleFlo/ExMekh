from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType

from qs.core.agents.capabilities.data import CapabilitiesData
from qs.core.agents.node import Node, NodeUserMessage
from qs.interfaces.event_handler import DeepAgentEventHandler
from qs.interfaces.provider import Provider

from .agent_response import AgentResponse


class DeepAgent:
    """Initializes a node hierarchy and delegates execution to its root."""

    def __init__(
        self,
        provider: Provider,
        node: Node,
        work_root: Path,
        on_stream: DeepAgentEventHandler | None = None,
    ) -> None:
        if not isinstance(provider, Provider):
            raise TypeError("provider must be a Provider")

        if not isinstance(node, Node):
            raise TypeError("node must be a Node")
        work_root = self._validate_work_root(work_root)
        if on_stream is not None:
            DeepAgentEventHandler.validate_instance(on_stream)

        self._root_node = node
        self._capabilities_by_node = self._initialize_node_tree(
            provider,
            node,
            work_root,
            on_stream,
        )

    @property
    def root_node(self) -> Node:
        return self._root_node

    @property
    def capabilities_by_node(self) -> Mapping[Node, CapabilitiesData]:
        return self._capabilities_by_node

    def run(self, user_message: NodeUserMessage) -> AgentResponse:
        return self._root_node.run(user_message)

    def get_tools_text(self) -> str:
        """Return the root's tools JSON exactly as serialized for its provider.

        Includes contextual skill descriptions and runtime tools. Does not call
        inference or change history. The server's rendered prompt is separate.
        Providers must implement tools request serialization.
        """
        return self._root_node.get_tools_text()

    def steer(self, user_message: NodeUserMessage) -> None:
        """Queue guidance for the active root run without waiting for its response."""
        self._root_node.steer(user_message)

    @classmethod
    def _initialize_node_tree(
        cls,
        provider: Provider,
        root_node: Node,
        work_root: Path,
        on_stream: DeepAgentEventHandler | None,
    ) -> MappingProxyType[Node, CapabilitiesData]:
        """Recursive function to initialize the node tree.

        Args:
            provider (Provider): The provider to initialize the nodes with.
            root_node (Node): The root node of the hierarchy.

        Raises:
            ValueError: If the node hierarchy contains cycles.

        Returns:
            MappingProxyType[Node, CapabilitiesData]: A read-only mapping of nodes to their capabilities data.
        """
        capabilities_by_node: dict[Node, CapabilitiesData] = {}
        active_nodes: set[Node] = set()

        def initialize(node: Node) -> None:
            if node in active_nodes:
                raise ValueError("Node hierarchy cannot contain cycles")

            if node in capabilities_by_node:
                return

            active_nodes.add(node)
            capabilities_by_node[node] = node.initialize(
                provider,
                work_root=work_root,
                on_stream=on_stream,
            )
            for child in node.children:
                initialize(child)
            node.update_sub_agent_tool_description()
            active_nodes.remove(node)

        initialize(root_node)
        cls._validate_capabilities_by_node(capabilities_by_node)
        return MappingProxyType(capabilities_by_node)

    @staticmethod
    def _validate_capabilities_by_node(capabilities_by_node) -> None:
        context_ids = [node.agent_context_id for node in capabilities_by_node]
        if len(context_ids) != len(set(context_ids)):
            duplicates = sorted(
                cid
                for cid in set(context_ids)
                if cid is not None and context_ids.count(cid) > 1
            )
            raise ValueError(
                "Distinct nodes in hierarchy must have unique agent_context_id: "
                + ", ".join(duplicates)
            )

    @staticmethod
    def _validate_work_root(work_root: Path) -> Path:
        if not isinstance(work_root, Path):
            raise TypeError("work_root must be a Path")
        if not work_root.exists():
            raise FileNotFoundError(f"work_root does not exist: {work_root}")
        if not work_root.is_dir():
            raise ValueError(f"work_root must be a directory: {work_root}")
        if not os.access(work_root, os.R_OK | os.W_OK | os.X_OK):
            raise PermissionError(
                f"work_root must be readable, writable, and traversable: {work_root}"
            )
        return work_root.resolve()
