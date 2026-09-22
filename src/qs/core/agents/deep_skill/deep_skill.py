from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from qs.core.agents.capabilities.resource_type import ResourceType
from qs.core.agents.capability.markdown_capability import MarkdownCapability
from qs.core.agents.deep_skill.catalog import SkillCatalog
from qs.core.agents.deep_skill.metadata import DeepSkillMetadata
from qs.core.models.assistant_message import ToolCall
from qs.core.models.tool_message import ToolMessage

if TYPE_CHECKING:
    from qs.core.agents.deep_skill.execution import DeepSkillExecutionTracker


class DeepSkill(MarkdownCapability[DeepSkillMetadata]):
    resource_type = ResourceType.DEEP_SKILL

    def __init__(
        self,
        *,
        name: str,
        markdown_path: Path,
        metadata: DeepSkillMetadata,
        body: str,
        references: tuple[str, ...],
    ) -> None:
        if not isinstance(metadata, DeepSkillMetadata):
            raise TypeError("metadata must be DeepSkillMetadata")

        self._catalog: SkillCatalog | None = None
        super().__init__(
            name=name,
            markdown_path=markdown_path,
            metadata=metadata,
            body=body,
            references=references,
        )

    @property
    def catalog(self) -> SkillCatalog | None:
        return self._catalog

    def render_instructions(self, exit_instructions: str) -> str:
        return f"{self._render_body()}\n\n{exit_instructions}"

    def inherits_parent_tools(self) -> bool:
        return self.metadata.get_inherits_parent_tools()

    def _render_body(self) -> str:
        return self._render_with_catalog(self.body)

    def _render_with_catalog(self, body: str) -> str:
        body = body.strip()
        if self._catalog is None:
            return body
        return f"{body}\n\n{self._catalog.render_prompt_section()}"

    def attach_catalog(self, catalog: SkillCatalog) -> None:
        if not isinstance(catalog, SkillCatalog):
            raise TypeError("catalog must be SkillCatalog")

        if self._catalog is not None:
            raise RuntimeError(f"Catalog already attached to deep skill '{self.name}'")

        self._catalog = catalog

    def __str__(self) -> str:
        return f"Name: {self.name}\nType: DeepSkill\nBody: {self.body[:50]}...\nDescription: {self.description[:50]}..."


class CalledDeepSkill:
    def __init__(self, deep_skill: DeepSkill, tool_call: ToolCall) -> None:
        if not isinstance(deep_skill, DeepSkill):
            raise TypeError("deep_skill must be DeepSkill")
        if not isinstance(tool_call, ToolCall):
            raise TypeError("tool_call must be ToolCall")

        self._deep_skill = deep_skill
        self._tool_call = tool_call

        self._validate_matching()

    def _validate_matching(self) -> None:
        if self._tool_call.get_name() != self._deep_skill.name:
            raise ValueError(
                f"Tool call name '{self._tool_call.get_name()}' does not match "
                f"deep skill name '{self._deep_skill.name}'"
            )

    @property
    def name(self) -> str:
        return self._deep_skill.name

    @property
    def call_id(self) -> str:
        return self._tool_call.get_id()

    @property
    def arguments(self) -> dict[str, Any]:
        return self._tool_call.get_arguments()

    @property
    def tool_call(self) -> ToolCall:
        return self._tool_call

    def render_instructions(self, exit_instructions: str) -> str:
        return self._deep_skill.render_instructions(exit_instructions)

    def open_execution(
        self,
        tracker: DeepSkillExecutionTracker,
        start_index: int,
        thinking: str | None = None,
        preceding_tool_responses: tuple[ToolMessage, ...] = (),
    ) -> None:
        tracker.open(
            self._deep_skill,
            self.call_id,
            start_index,
            self.arguments,
            thinking,
            preceding_tool_responses,
        )
