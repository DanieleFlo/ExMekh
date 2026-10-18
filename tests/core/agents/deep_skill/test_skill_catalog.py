from pathlib import Path
from xml.etree import ElementTree

import pytest

from exmekh_lib.core.agents.deep_skill.catalog import (
    SkillCatalog,
    SkillCatalogEntry,
)
from exmekh_lib.core.agents.deep_skill.deep_skill import DeepSkill
from exmekh_lib.core.agents.deep_skill.metadata import DeepSkillMetadata
from exmekh_lib.core.models.json_schema import SchemaDescription
from exmekh_lib.utils.xml import read_text_element


def test_parameter_documentation_uses_lossless_xml_text():
    text = "Use <value> & preserve\rnewlines\x00."
    catalog = SkillCatalog(
        sub_tools=(
            SkillCatalogEntry(
                name="search",
                description="Search.",
                parameters=(SchemaDescription('/properties/a<&"', text),),
            ),
        )
    )
    root = ElementTree.fromstring(catalog.render_prompt_section())
    parameter = root.find("./parameter-descriptions/tool/parameter")
    assert parameter is not None
    assert parameter.attrib["path"] == '/properties/a<&"'
    assert parameter.attrib["encoding"] == "base64"
    assert read_text_element(parameter) == text


@pytest.mark.parametrize("text", ["Preserve\rcarriage returns.", "Keep\x00controls."])
def test_tool_descriptions_use_the_same_lossless_xml_encoding(text):
    catalog = SkillCatalog(sub_tools=(SkillCatalogEntry("search", text),))

    root = ElementTree.fromstring(catalog.render_prompt_section())

    assert read_text_element(root.find("./sub-tools/tool")) == text


def _exit_instructions() -> str:
    return '<exit-tool name="finish">Finish instructions</exit-tool>'


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

    root = ElementTree.fromstring(catalog.render_prompt_section())
    assert root.findtext("./sub-skills/skill") == "Research the requested topic."
    assert root.find("./sub-skills/skill").get("name") == "research"
    assert root.findtext("./sub-tools/tool") == "Search indexed documents."
    assert root.find("./sub-tools/tool").get("name") == "search"


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

    assert skill.render_instructions(_exit_instructions()) == (
        "Body\n\n<catalog />\n\n"
        '<exit-tool name="finish">Finish instructions</exit-tool>'
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
    assert skill.render_instructions(_exit_instructions()) == (
        f"Body\n\n{catalog.render_prompt_section()}\n\n{_exit_instructions()}"
    )
