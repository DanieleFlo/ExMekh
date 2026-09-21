from __future__ import annotations

from qs.core.agents.deep_skill.deep_skill import DeepSkill
from qs.core.agents.hds.hds import HDS


class AgenticScope:
    """Build provider capability scopes from the active skill path."""

    def __init__(
        self,
        hds: HDS,
        exit_tool_name: str,
    ) -> None:
        self._hds = hds
        self._exit_tool_name = exit_tool_name
        self._pending_return_skill_call_id: str | None = None

    @property
    def node_id(self) -> str:
        return self._hds.get_name()

    def build_extra(
        self,
        active_skills: tuple[DeepSkill, ...],
        depth: int,
    ) -> dict[str, dict[str, object]]:
        capability = active_skills[-1] if active_skills else self._hds
        allowed_tools = self._get_allowed_tool_names(active_skills)
        allowed_skills = list(capability.get_sub_skill_names())

        if depth > 0:
            allowed_tools.append(self._exit_tool_name)

        agentic: dict[str, object] = {
            "allowed_tools": allowed_tools,
            "allowed_skills": allowed_skills,
        }
        if self._pending_return_skill_call_id is not None:
            agentic["operation"] = "return"
            agentic["skill_call_id"] = self._pending_return_skill_call_id

        return {"agentic": agentic}

    def register_skill_return(self, skill_call_id: str) -> None:
        self._pending_return_skill_call_id = skill_call_id

    def acknowledge_extra(self) -> None:
        self._pending_return_skill_call_id = None

    def _get_allowed_tool_names(
        self,
        active_skills: tuple[DeepSkill, ...],
    ) -> list[str]:
        allowed_tools = list(self._hds.get_sub_tool_names())
        for skill in active_skills:
            local_tools = list(skill.get_sub_tool_names())
            if skill.inherits_parent_tools():
                local_tools.extend(allowed_tools)
            allowed_tools = list(dict.fromkeys(local_tools))
        return allowed_tools
