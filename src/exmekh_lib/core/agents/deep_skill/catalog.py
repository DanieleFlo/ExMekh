from __future__ import annotations

from dataclasses import dataclass
from xml.etree import ElementTree as ET

from exmekh_lib.config.capabilities import CAPABILITY_FORMAT
from exmekh_lib.core.models.json_schema import SchemaDescription
from exmekh_lib.utils.xml import append_text_element, render_xml


@dataclass(frozen=True, slots=True)
class SkillCatalogEntry:
    name: str
    description: str
    parameters: tuple[SchemaDescription, ...] = ()


@dataclass(frozen=True, slots=True)
class SkillCatalog:
    sub_skills: tuple[SkillCatalogEntry, ...] = ()
    sub_tools: tuple[SkillCatalogEntry, ...] = ()
    _TAG = CAPABILITY_FORMAT.catalog_tag

    def render_prompt_section(self) -> str:
        root = ET.Element(self._TAG)
        self._append_section(root, "sub-skills", "skill", self.sub_skills)
        self._append_section(root, "sub-tools", "tool", self.sub_tools)
        self._append_parameters(root)
        return render_xml(root)

    def _append_parameters(self, catalog: ET.Element) -> None:
        entries = tuple(
            entry for entry in (*self.sub_skills, *self.sub_tools) if entry.parameters
        )
        if not entries:
            return
        root = ET.SubElement(
            catalog, "parameter-descriptions", {"path-format": "json-pointer"}
        )
        for entry in entries:
            tool = ET.SubElement(root, "tool", {"name": entry.name})
            for parameter in entry.parameters:
                append_text_element(
                    tool,
                    "parameter",
                    parameter.text,
                    attributes={"path": parameter.path, "type": "string"},
                )

    @staticmethod
    def _append_section(
        catalog: ET.Element,
        tag: str,
        entry_tag: str,
        entries: tuple[SkillCatalogEntry, ...],
    ) -> None:
        if not entries:
            return
        section = ET.SubElement(catalog, tag)
        for entry in entries:
            append_text_element(
                section, entry_tag, entry.description, attributes={"name": entry.name}
            )
