from pathlib import Path
from xml.etree import ElementTree

import pytest

from qs.core.agents.deep_skill.catalog import (
    SkillCatalog,
    SkillCatalogEntry,
)
from qs.core.agents.deep_skill.deep_skill import DeepSkill
from qs.core.agents.deep_skill.metadata import DeepSkillMetadata
from qs.core.agents.exit_skill.exit_skill import ExitSkill


def _exit_skill(tmp_path: Path) -> ExitSkill:
    return ExitSkill(
        name="finish",
        markdown_path=tmp_path / "finish.md",
        metadata=DeepSkillMetadata(
            description="Finish",
            sub_skills=(),
            sub_tools=(),
            inherits_parent_tools=False,
        ),
        body="Finish instructions",
        references=(),
    )


def test_render_prompt_section_includes_descriptions() -> None:
    catalog = SkillCatalog(
        sub_skills=(
            SkillCatalogEntry(
                name="research",
                description="Research the requested topic.",
            ),
        ),
        sub_tools=(
            SkillCatalogEntry(
                name="search",
                description="Search indexed documents.",
            ),
        ),
    )

    assert catalog.render_prompt_section() == (
        "<catalog><sub-skills>\n"
        '<skill name="research">Research the requested topic.</skill>\n'
        "</sub-skills>\n\n"
        "<sub-tools>\n"
        '<tool name="search">Search indexed documents.</tool>\n'
        "</sub-tools></catalog>"
    )


def test_render_prompt_section_escapes_xml_content() -> None:
    catalog = SkillCatalog(
        sub_tools=(
            SkillCatalogEntry(
                name='search&"docs',
                description='Search <docs> & "references".',
            ),
        )
    )

    root = ElementTree.fromstring(catalog.render_prompt_section())
    tool = root.find("./sub-tools/tool")

    assert tool is not None
    assert tool.attrib == {"name": 'search&"docs'}
    assert tool.text == 'Search <docs> & "references".'


def test_attached_empty_catalog_renders_empty_prompt_section(tmp_path: Path) -> None:
    skill = DeepSkill(
        name="skill",
        markdown_path=tmp_path / "skill.md",
        metadata=DeepSkillMetadata(
            description="Description",
            inherits_parent_tools=False,
            sub_skills=(),
            sub_tools=(),
        ),
        body="Body",
        references=(),
    )

    skill.attach_catalog(SkillCatalog())

    assert skill.render_instructions(_exit_skill(tmp_path)) == (
        "Body\n\n<catalog></catalog>\n\n"
        '<exit-skill tools="">Finish instructions</exit-skill>'
    )

    with pytest.raises(RuntimeError, match="Catalog already attached"):
        skill.attach_catalog(SkillCatalog())


def test_attached_catalog_renders_prompt_section(tmp_path: Path) -> None:
    skill = DeepSkill(
        name="skill",
        markdown_path=tmp_path / "skill.md",
        metadata=DeepSkillMetadata(
            description="Description",
            inherits_parent_tools=False,
            sub_skills=("child",),
            sub_tools=(),
        ),
        body="Body",
        references=(),
    )
    catalog = SkillCatalog(
        sub_skills=(SkillCatalogEntry("child", "Child description"),)
    )

    skill.attach_catalog(catalog)

    assert skill.catalog is catalog
    assert skill.render_instructions(_exit_skill(tmp_path)) == (
        "Body\n\n<catalog><sub-skills>\n"
        '<skill name="child">Child description</skill>\n'
        "</sub-skills></catalog>\n\n"
        '<exit-skill tools="">Finish instructions</exit-skill>'
    )
