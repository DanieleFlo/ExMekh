from pathlib import Path
from typing import cast

import pytest

from qs.core.agents.deep_skill.catalog import SkillCatalog
from qs.core.agents.deep_skill.deep_skill import CalledDeepSkill, DeepSkill
from qs.core.agents.deep_skill.metadata import DeepSkillMetadata
from qs.core.models.assistant_message import FunctionCall, ToolCall


def _metadata() -> DeepSkillMetadata:
    return DeepSkillMetadata(
        description="Description",
        inherits_parent_tools=False,
        sub_skills=(),
        sub_tools=(),
    )


@pytest.mark.parametrize(
    ("markdown_path", "metadata", "body", "references", "error", "message"),
    [
        (
            cast(Path, "skill.md"),
            _metadata(),
            "Body",
            (),
            TypeError,
            "markdown_path must be a Path",
        ),
        (
            Path("skill.md"),
            cast(DeepSkillMetadata, object()),
            "Body",
            (),
            TypeError,
            "metadata must be DeepSkillMetadata",
        ),
        (
            Path("skill.md"),
            _metadata(),
            cast(str, object()),
            (),
            TypeError,
            "body must be a string",
        ),
        (
            Path("skill.md"),
            _metadata(),
            "Body",
            cast(tuple[str, ...], []),
            TypeError,
            "references must be a tuple",
        ),
        (
            Path("skill.md"),
            _metadata(),
            "Body",
            (cast(str, object()),),
            TypeError,
            "references must contain only strings",
        ),
        (
            Path("skill.md"),
            _metadata(),
            "Body",
            ("child", "child"),
            ValueError,
            "references cannot contain duplicates",
        ),
    ],
)
def test_rejects_invalid_constructor_inputs(
    markdown_path: Path,
    metadata: DeepSkillMetadata,
    body: str,
    references: tuple[str, ...],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        DeepSkill(
            name="skill",
            markdown_path=markdown_path,
            metadata=metadata,
            body=body,
            references=references,
        )


def test_rejects_non_catalog_attachment(tmp_path: Path) -> None:
    skill = DeepSkill(
        name="skill",
        markdown_path=tmp_path / "skill.md",
        metadata=_metadata(),
        body="Body",
        references=(),
    )

    with pytest.raises(TypeError, match="catalog must be SkillCatalog"):
        skill.attach_catalog(cast(SkillCatalog, object()))


def test_gets_direct_tool_names(tmp_path: Path) -> None:
    skill = DeepSkill(
        name="skill",
        markdown_path=tmp_path / "skill.md",
        metadata=DeepSkillMetadata(
            description="Description",
            inherits_parent_tools=False,
            sub_skills=("child",),
            sub_tools=("search",),
        ),
        body="Body",
        references=(),
    )

    assert skill.get_direct_tool_names() == {"child", "search"}


def test_called_deep_skill_validates_types_and_name(tmp_path: Path) -> None:
    skill = DeepSkill(
        name="skill",
        markdown_path=tmp_path / "skill.md",
        metadata=_metadata(),
        body="Body",
        references=(),
    )
    matching_call = ToolCall("call-1", FunctionCall("skill", {}))

    assert CalledDeepSkill(skill, matching_call).name == "skill"

    with pytest.raises(TypeError, match="deep_skill must be DeepSkill"):
        CalledDeepSkill(cast(DeepSkill, object()), matching_call)
    with pytest.raises(TypeError, match="tool_call must be ToolCall"):
        CalledDeepSkill(skill, cast(ToolCall, object()))
    with pytest.raises(ValueError, match="does not match deep skill name"):
        CalledDeepSkill(
            skill,
            ToolCall("call-2", FunctionCall("other", {})),
        )
