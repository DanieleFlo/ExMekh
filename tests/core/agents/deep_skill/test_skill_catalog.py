from pathlib import Path

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


def test_render_includes_sub_skill_and_sub_tool_descriptions() -> None:
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

    assert catalog.render() == (
        "## Available sub-skills\n"
        "- **research**: Research the requested topic.\n\n"
        "## Available sub-tools\n"
        "- **search**: Search indexed documents."
    )


def test_attached_empty_catalog_does_not_change_skill_body(tmp_path: Path) -> None:
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

    assert skill.get_skill(_exit_skill(tmp_path)) == (
        "Body\n\n<catalog></catalog>\n\n<exit-skill>Finish instructions</exit-skill>"
    )

    with pytest.raises(RuntimeError, match="Catalog already attached"):
        skill.attach_catalog(SkillCatalog())


def test_attached_catalog_is_appended_to_skill_body(tmp_path: Path) -> None:
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
    assert skill.get_skill(_exit_skill(tmp_path)) == (
        "Body\n\n<catalog>## Available sub-skills\n"
        "- **child**: Child description</catalog>\n\n"
        "<exit-skill>Finish instructions</exit-skill>"
    )
