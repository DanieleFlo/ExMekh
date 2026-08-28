from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from qs.core.agents.capabilities.resource_type import ResourceType
from qs.core.agents.capability.markdown_capability import MarkdownCapability
from qs.core.agents.deep_skill.catalog import SkillCatalog
from qs.core.agents.deep_skill.metadata import DeepSkillMetadata
from qs.core.models.assistant_message import ToolCall

if TYPE_CHECKING:
    from qs.core.agents.exit_skill.exit_skill import ExitSkill


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

    def get_skill(self, exit_skill: ExitSkill) -> str:
        body = self._render_body()
        body_with_exit_skill = self._append_exit_skill(body, exit_skill)
        return body_with_exit_skill

    def _render_body(self) -> str:
        body = self.body.strip()
        body = self._append_catalog(body)
        return body

    def _append_catalog(self, body: str) -> str:
        body = body.strip()
        if self._catalog is None:
            return body
        catalog_rendered = self._catalog.render()
        return f"{body}\n\n<{SkillCatalog.TAG}>{catalog_rendered}</{SkillCatalog.TAG}>"

    def _append_exit_skill(self, body: str, exit_skill: ExitSkill) -> str:
        body = body.strip()
        exit_skill_rendered = exit_skill.body.strip()
        return f"{body}\n\n<{exit_skill.TAG}>{exit_skill_rendered}</{exit_skill.TAG}>"

    def attach_catalog(self, catalog: SkillCatalog) -> None:
        if not isinstance(catalog, SkillCatalog):
            raise TypeError("catalog must be SkillCatalog")

        if self._catalog is not None:
            raise RuntimeError(f"Catalog already attached to deep skill '{self.name}'")

        self._catalog = catalog


class CalledDeepSkill:
    def __init__(self, deep_skill: DeepSkill, tool_call: ToolCall) -> None:
        if not isinstance(deep_skill, DeepSkill):
            raise TypeError("deep_skill must be DeepSkill")
        if not isinstance(tool_call, ToolCall):
            raise TypeError("tool_call must be ToolCall")

        self.deep_skill = deep_skill
        self.tool_call = tool_call

        self.validate_matching()

    def validate_matching(self) -> None:
        if self.tool_call.get_name() != self.deep_skill.name:
            raise ValueError(
                f"Tool call name '{self.tool_call.get_name()}' does not match deep skill name '{self.deep_skill.name}'"
            )

    @property
    def name(self) -> str:
        return self.deep_skill.name
