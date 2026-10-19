from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from exmekh_lib.config.agent_text import AGENT_TEXT
from exmekh_lib.config.capabilities import CAPABILITY_FORMAT
from exmekh_lib.core.agents.capabilities.resource_type import ResourceType
from exmekh_lib.core.agents.capability.conditions import CapabilityPromptContext
from exmekh_lib.core.agents.capability.markdown_capability import MarkdownCapability
from exmekh_lib.core.agents.deep_skill.catalog import SkillCatalog
from exmekh_lib.core.agents.deep_skill.metadata import DeepSkillMetadata
from exmekh_lib.core.models.assistant_message import ToolCall
from exmekh_lib.core.models.tool_message import ToolMessage

if TYPE_CHECKING:
    from exmekh_lib.core.agents.deep_skill.execution import DeepSkillExecutionTracker


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
            raise TypeError(AGENT_TEXT.metadata_must_be_deepskillmetadata)

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

    def render_instructions(
        self, exit_instructions: str, *, context: CapabilityPromptContext | None = None
    ) -> str:
        return f"{self._render_with_catalog(self.render_body(context))}\n\n{exit_instructions}"

    def inherits_parent_tools(self) -> bool:
        return self.metadata.get_inherits_parent_tools()

    def _render_body(self) -> str:
        return self._render_with_catalog(self.render_body())

    def _render_with_catalog(self, body: str) -> str:
        body = body.strip()
        if self._catalog is None:
            return body
        return f"{body}\n\n{self._catalog.render_prompt_section()}"

    def attach_catalog(self, catalog: SkillCatalog) -> None:
        if not isinstance(catalog, SkillCatalog):
            raise TypeError(AGENT_TEXT.catalog_must_be_skillcatalog)

        if self._catalog is not None:
            raise RuntimeError(
                AGENT_TEXT.catalog_already_attached_to_deep_skill.format(name=self.name)
            )

        self._catalog = catalog

    def __str__(self) -> str:
        return AGENT_TEXT.name_type_deepskill_body_description.format(
            name=self.name,
            body=self.body[: CAPABILITY_FORMAT.representation_preview_chars],
            description=self.description[
                : CAPABILITY_FORMAT.representation_preview_chars
            ],
        )


class CalledDeepSkill:
    def __init__(self, deep_skill: DeepSkill, tool_call: ToolCall) -> None:
        if not isinstance(deep_skill, DeepSkill):
            raise TypeError(AGENT_TEXT.deep_skill_must_be_deepskill)
        if not isinstance(tool_call, ToolCall):
            raise TypeError(AGENT_TEXT.tool_call_must_be_toolcall)

        self._deep_skill = deep_skill
        self._tool_call = tool_call

        self._validate_matching()

    def _validate_matching(self) -> None:
        if self._tool_call.get_name() != self._deep_skill.get_name():
            raise ValueError(
                AGENT_TEXT.tool_call_name_does_not_match_deep_skill_name.format(
                    tool_call_get_name=self._tool_call.get_name(),
                    deep_skill_get_name=self._deep_skill.get_name(),
                )
            )

    @property
    def name(self) -> str:
        return self._deep_skill.get_name()

    @property
    def call_id(self) -> str:
        return self._tool_call.get_id()

    @property
    def arguments(self) -> dict[str, Any]:
        return self._tool_call.get_arguments()

    @property
    def tool_call(self) -> ToolCall:
        return self._tool_call

    def render_instructions(
        self, exit_instructions: str, *, context: CapabilityPromptContext | None = None
    ) -> str:
        return self._deep_skill.render_instructions(exit_instructions, context=context)

    def open_execution(
        self,
        tracker: DeepSkillExecutionTracker,
        start_index: int,
        thinking: str | None = None,
        preceding_tool_responses: tuple[ToolMessage, ...] = (),
        opening_history_position: int = 0,
    ) -> None:
        tracker.open(
            self._deep_skill,
            self.call_id,
            start_index,
            self.arguments,
            thinking,
            preceding_tool_responses,
            opening_history_position,
        )
