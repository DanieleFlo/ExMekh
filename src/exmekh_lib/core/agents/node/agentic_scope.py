from __future__ import annotations

from exmekh_lib.config.runtime_text import RUNTIME_TEXT
from exmekh_lib.core.agents.deep_skill.deep_skill import DeepSkill
from exmekh_lib.core.agents.hds.hds import HDS
from exmekh_lib.core.models.assistant_message import ToolCall


class AgenticScope:
    """Build provider capability scopes from the active skill path."""

    def __init__(
        self,
        hds: HDS,
        exit_tool_name: str,
        runtime_tool_names: tuple[str, ...] = (),
        agent_context_id: str | None = None,
    ) -> None:
        self._hds = hds
        self._exit_tool_name = exit_tool_name
        self._runtime_tool_names = runtime_tool_names
        self._agent_context_id = self._resolved_context_id(agent_context_id, hds)
        self._validate_resolved_context_id()
        self._pending_return_skill_call_id: str | None = None

    @staticmethod
    def _resolved_context_id(agent_context_id: str | None, hds: HDS) -> str:
        return (
            agent_context_id
            if agent_context_id is not None
            else hds.get_agent_context_id()
        )

    def _validate_resolved_context_id(self) -> None:
        if not isinstance(self._agent_context_id, str):
            raise TypeError(RUNTIME_TEXT.agent_context_id_must_be_a_string)
        if not self._agent_context_id.strip():
            raise ValueError(RUNTIME_TEXT.agent_context_id_cannot_be_empty)

    @property
    def agent_context_id(self) -> str:
        return self._agent_context_id

    @property
    def node_id(self) -> str:
        return self._hds.get_name()

    @property
    def pending_return_call_id(self) -> str | None:
        return self._pending_return_skill_call_id

    def build_extra(
        self,
        active_skills: tuple[DeepSkill, ...],
        depth: int,
    ) -> dict[str, dict[str, object]]:
        capability = active_skills[-1] if active_skills else self._hds
        allowed_tools = self.get_allowed_tool_names(active_skills, depth)
        allowed_skills = list(capability.get_sub_skill_names())

        agentic: dict[str, object] = {
            "agent_context_id": self._agent_context_id,
            "allowed_tools": allowed_tools,
            "allowed_skills": allowed_skills,
        }
        if self._pending_return_skill_call_id is not None:
            agentic["operation"] = "return"
            agentic["skill_call_id"] = self._pending_return_skill_call_id

        return {"agentic": agentic}

    def validate_tool_calls(
        self,
        calls: list[ToolCall],
        active_skills: tuple[DeepSkill, ...],
        depth: int,
    ) -> None:
        """Check calls against the same capabilities advertised to the provider."""
        capability = active_skills[-1] if active_skills else self._hds
        allowed_tools = self.get_allowed_tool_names(active_skills, depth)
        allowed_skills = list(capability.get_sub_skill_names())
        allowed = set(allowed_tools) | set(allowed_skills)
        for call in calls:
            if call.get_name() not in allowed:
                skill_path = "/" + "/".join(skill.get_name() for skill in active_skills)
                raise ValueError(
                    RUNTIME_TEXT.capability_is_not_allowed_at_allowed_tools_allowed_skills.format(
                        call_get_name=call.get_name(),
                        skill_path=skill_path,
                        allowed_tools=allowed_tools,
                        allowed_skills=allowed_skills,
                    )
                )

    def register_skill_return(self, skill_call_id: str) -> None:
        self._pending_return_skill_call_id = skill_call_id

    def acknowledge_extra(self) -> None:
        self._pending_return_skill_call_id = None

    def get_allowed_tool_names(
        self,
        active_skills: tuple[DeepSkill, ...],
        depth: int,
    ) -> list[str]:
        allowed_tools = list(self._hds.get_sub_tool_names())
        for skill in active_skills:
            local_tools = list(skill.get_sub_tool_names())
            if skill.inherits_parent_tools():
                local_tools.extend(allowed_tools)
            allowed_tools = list(dict.fromkeys(local_tools))
        allowed_tools.extend(self._runtime_tool_names)
        if depth > 0:
            allowed_tools.append(self._exit_tool_name)
        return list(dict.fromkeys(allowed_tools))
