from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from qs.core.agents.capabilities.data import CapabilitiesData
from qs.core.agents.node import Node, NodeUserMessage
from qs.interfaces.provider import Provider

from .agent_response import AgentResponse
from .stream_event import DeepAgentEventHandler


class DeepAgent:
    """Initializes a node hierarchy and delegates execution to its root."""

    def __init__(
        self,
        provider: Provider,
        node: Node,
        on_stream: DeepAgentEventHandler | None = None,
    ) -> None:
        if not isinstance(provider, Provider):
            raise TypeError("provider must be a Provider")

        if not isinstance(node, Node):
            raise TypeError("node must be a Node")
        if on_stream is not None and not callable(on_stream):
            raise TypeError("on_stream must be a callable if provided")

        self._root_node = node
        self._capabilities_by_node = self._initialize_node_tree(
            provider,
            node,
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

    @staticmethod
    def _initialize_node_tree(
        provider: Provider,
        root_node: Node,
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
            capabilities_by_node[node] = node.initialize(provider, on_stream=on_stream)
            for child in node.children:
                initialize(child)
            active_nodes.remove(node)

        initialize(root_node)
        return MappingProxyType(capabilities_by_node)
