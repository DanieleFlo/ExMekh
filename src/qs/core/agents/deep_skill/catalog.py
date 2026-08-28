from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SkillCatalogEntry:
    name: str
    description: str


@dataclass(frozen=True, slots=True)
class SkillCatalog:
    sub_skills: tuple[SkillCatalogEntry, ...] = ()
    sub_tools: tuple[SkillCatalogEntry, ...] = ()
    _TAG = "catalog"

    def render_prompt_section(self) -> str:
        content = self._render_content()
        return f"<{self._TAG}>{content}</{self._TAG}>"

    def _render_content(self) -> str:
        sections = tuple(
            section
            for section in (
                self._render_section(
                    "Available sub-skills",
                    self.sub_skills,
                ),
                self._render_section(
                    "Available sub-tools",
                    self.sub_tools,
                ),
            )
            if section
        )

        return "\n\n".join(sections)

    @staticmethod
    def _render_section(
        title: str,
        entries: tuple[SkillCatalogEntry, ...],
    ) -> str:
        if not entries:
            return ""

        items = "\n".join(
            f"- **{entry.name}**: {entry.description}" for entry in entries
        )

        return f"## {title}\n{items}"
