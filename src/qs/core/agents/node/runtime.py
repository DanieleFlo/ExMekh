from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final

from qs.core.agents.capabilities.data import CapabilitiesData
from qs.core.agents.deep_agent.agent_response import AgentResponse
from qs.core.agents.exit_skill.exit_skill import ExitSkill
from qs.core.agents.hds.hds import HDS
from qs.core.models.client import Client
from qs.core.models.client_request import ClientRequest
from qs.core.models.provider_response import ProviderResponse
from qs.core.models.system_message import SystemMessage
from qs.core.models.tools import Tools
from qs.core.models.user_message import ErrorUserMessage, UserMessage
from qs.interfaces.provider import Provider

from .agentic_scope import AgenticScope
from .tool_call_executor import ToolCallExecutor
from .tool_call_resolver import ToolCallResolver
from .types import NodeUserMessage

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class NodeRuntimeConfig:
    provider: Provider
    capabilities: CapabilitiesData
    hds: HDS
    exit_skill: ExitSkill
    persistent_history: bool
    initial_depth: int

    def build_tools(self) -> Tools:
        return Tools(list(self.capabilities.get_contextual_tools(self.hds)))

    def build_executor(self, client: Client) -> ToolCallExecutor:
        agentic_scope = AgenticScope(
            hds=self.hds,
            exit_skill_name=self.exit_skill.name,
            exit_tool_names=self.capabilities.get_exit_tool_names(self.exit_skill.name),
        )
        return ToolCallExecutor(
            client=client,
            exit_skill=self.exit_skill,
            initial_depth=self.initial_depth,
            agentic_scope=agentic_scope,
        )

    def get_hds_name(self) -> str:
        return self.hds.name

    def build_system_message(self) -> SystemMessage:
        return SystemMessage(content=self.hds.get_hds())


class NodeRuntime:
    """Own the conversational lifecycle of an initialized node."""

    REQUIRED_TOOL_CHOICE: Final[str] = "required"
    EMPTY_RESPONSE_MESSAGE: Final[str] = (
        "A response without a tool call and without content is not allowed. "
        "Respond with either a tool call or non-empty content."
    )

    def __init__(self, config: NodeRuntimeConfig) -> None:
        self._config = config
        self._resolver = ToolCallResolver(config.capabilities)
        self._client: Client
        self._executor: ToolCallExecutor
        self._reset_session()

    @property
    def depth(self) -> int:
        return self._executor.depth

    def run(self, user_message: NodeUserMessage) -> AgentResponse:
        if not self._config.persistent_history:
            self._reset_session()

        logger.info(
            "Starting node run: node=%s depth=%d",
            self._config.get_hds_name(),
            self.depth,
        )
        response = self._client.call(
            ClientRequest(
                message=UserMessage(content=user_message),
                extra=self._executor.build_agentic_extra(),
                tool_choice=self._current_tool_choice(),
            )
        )
        response = self._route_until_complete(response)
        logger.info(
            "Completed node run: node=%s depth=%d root_executions=%d",
            self._config.get_hds_name(),
            self.depth,
            len(self._executor.skill_executions),
        )
        return AgentResponse(
            response=response,
            usage=response.usage,
            thinking=response.thinking,
            history=self._client.dump_history(),
            skill_executions=self._executor.skill_executions,
        )

    def _route_until_complete(self, response: ProviderResponse) -> ProviderResponse:
        while True:
            tool_calls = response.get_tool_calls()
            if tool_calls:
                resolved_tool_calls = self._resolver.resolve_all(tool_calls)
                self._executor.execute_all(resolved_tool_calls)

                tool_choice = self._current_tool_choice()
                response = self._client.continue_after_tools(
                    extra=self._executor.build_agentic_extra(),
                    tool_choice=tool_choice,
                )
                self._executor.acknowledge_agentic_extra()
                continue

            content = response.get_message()
            if isinstance(content, str) and content.strip():
                return response

            response = self._client.call(
                ClientRequest(
                    message=ErrorUserMessage(self.EMPTY_RESPONSE_MESSAGE),
                    extra=self._executor.build_agentic_extra(),
                    tool_choice=self._current_tool_choice(),
                )
            )

    def _reset_session(self) -> None:
        tools = self._config.build_tools()
        self._client = Client(
            provider=self._config.provider,
            system_message=self._config.build_system_message(),
            tools=tools,
        )

        self._executor = self._config.build_executor(self._client)

    def _current_tool_choice(self) -> str | None:
        if self.depth > 0:
            return self.REQUIRED_TOOL_CHOICE
        return None
