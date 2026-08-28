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
from qs.core.models.tool import Tool
from qs.core.models.tools import Tools
from qs.core.models.user_message import ErrorUserMessage, UserMessage
from qs.interfaces.provider import Provider

from .tool_call_executor import ToolCallExecutor
from .tool_call_resolver import ToolCallResolver
from .types import NodeUserMessage

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class NodeRuntimeConfig:
    provider: Provider
    capabilities: CapabilitiesData
    hds_name: str
    exit_skill: ExitSkill
    persistent_history: bool
    initial_depth: int

    def build_tools(self, hds: HDS) -> Tools:
        direct_tool_names = hds.get_direct_tool_names()
        skill_availability = self.capabilities.get_skill_availability()

        tools = [
            self._prepare_tool(tool, direct_tool_names, skill_availability)
            for tool in self.capabilities.get_all_tools()
        ]
        return Tools(tools)

    @staticmethod
    def _prepare_tool(
        tool: Tool,
        direct_tool_names: set[str],
        skill_availability: dict[str, list[str]],
    ) -> Tool:
        if tool.name in direct_tool_names:
            return tool

        skill_names = skill_availability.get(tool.name, [])
        if not skill_names:
            description = ""
        elif len(skill_names) == 1:
            description = f"Available in skill '{skill_names[0]}'."
        else:
            names = ", ".join(f"'{name}'" for name in skill_names)
            description = f"Available in skills {names}."

        return tool.with_description(description)


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
            "Starting node run: node=%s depth=%d", self._config.hds_name, self.depth
        )
        response = self._client.call(
            ClientRequest(
                message=UserMessage(content=user_message),
                tool_choice=self._current_tool_choice(),
            )
        )
        response = self._route_until_complete(response)
        logger.info(
            "Completed node run: node=%s depth=%d root_executions=%d",
            self._config.hds_name,
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
            if tool_calls is not None:
                self.validate_tool_calls(tool_calls)
                self._executor.execute_all(self._resolver.resolve_all(tool_calls))

                tool_choice = self._current_tool_choice()
                response = self._client.continue_after_tools(tool_choice=tool_choice)
                continue

            content = response.get_message()
            if isinstance(content, str) and content.strip():
                return response

            response = self._client.call(
                ClientRequest(
                    message=ErrorUserMessage(self.EMPTY_RESPONSE_MESSAGE),
                    tool_choice=self._current_tool_choice(),
                )
            )

    def _reset_session(self) -> None:
        hds = next(
            hds
            for hds in self._config.capabilities.hds
            if hds.name == self._config.hds_name
        )

        tools = self._config.build_tools(hds)
        content = hds.get_hds()
        system_message = SystemMessage(content=content)
        self._client = Client(
            provider=self._config.provider,
            system_message=system_message,
            tools=tools,
        )

        self._executor = ToolCallExecutor(
            client=self._client,
            exit_skill=self._config.exit_skill,
            initial_depth=self._config.initial_depth,
        )

    def _current_tool_choice(self) -> str | None:
        if self.depth > 0:
            return self.REQUIRED_TOOL_CHOICE
        return None

    @staticmethod
    def validate_tool_calls(tool_calls: object) -> None:
        if not isinstance(tool_calls, list):
            raise TypeError("tool_calls must be a list or None")
        if not tool_calls:
            raise ValueError("Tool calls list cannot be empty")
