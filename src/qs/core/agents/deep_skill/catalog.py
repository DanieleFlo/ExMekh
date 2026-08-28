from __future__ import annotations

from dataclasses import dataclass
from html import escape


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
                    "sub-skills",
                    "skill",
                    self.sub_skills,
                ),
                self._render_section(
                    "sub-tools",
                    "tool",
                    self.sub_tools,
                ),
            )
            if section
        )

        return "\n\n".join(sections)

    @staticmethod
    def _render_section(
        tag: str,
        entry_tag: str,
        entries: tuple[SkillCatalogEntry, ...],
    ) -> str:
        if not entries:
            return ""

        items = "\n".join(
            f'<{entry_tag} name="{escape(entry.name, quote=True)}">'
            f"{escape(entry.description)}</{entry_tag}>"
            for entry in entries
        )

        return f"<{tag}>\n{items}\n</{tag}>"
