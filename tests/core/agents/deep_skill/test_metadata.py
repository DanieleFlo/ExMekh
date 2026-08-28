from typing import Any, cast

import pytest

from qs.core.agents.deep_skill.metadata import (
    DeepSkillMetadata,
    DeepSkillMetadataParser,
)


def test_parses_and_normalizes_all_metadata_fields() -> None:
    metadata = DeepSkillMetadataParser().parse(
        {
            "description": "  Description  ",
            "inherits-parent-tools": True,
            "sub-skills": "  child  ",
            "sub-tools": [" search ", "write"],
        }
    )

    assert metadata == DeepSkillMetadata(
        description="Description",
        inherits_parent_tools=True,
        sub_skills=("child",),
        sub_tools=("search", "write"),
    )


def test_defaults_inherits_parent_tools_to_false() -> None:
    metadata = DeepSkillMetadataParser().parse(
        {
            "description": "Description",
        }
    )

    assert metadata.inherits_parent_tools is False


@pytest.mark.parametrize(
    ("data", "error", "message"),
    [
        (cast(dict[str, Any], []), TypeError, "data must be a dictionary"),
        (
            {"description": "Skill", "extra": 1},
            ValueError,
            "Unknown metadata fields: extra",
        ),
        ({}, ValueError, "Missing required metadata fields"),
        (
            {"description": 1},
            TypeError,
            "field 'description' must be a string",
        ),
        (
            {"description": "  "},
            ValueError,
            "field 'description' cannot be empty",
        ),
        (
            {
                "description": "Skill",
                "inherits-parent-tools": 1,
            },
            TypeError,
            "field 'inherits-parent-tools' must be a boolean",
        ),
        (
            {
                "description": "Skill",
                "sub-skills": 1,
            },
            TypeError,
            "field 'sub-skills' must be a string or a list",
        ),
        (
            {
                "description": "Skill",
                "sub-skills": [1],
            },
            TypeError,
            "field 'sub-skills' must contain only strings",
        ),
        (
            {
                "description": "Skill",
                "sub-skills": ["  "],
            },
            ValueError,
            "field 'sub-skills' cannot contain empty values",
        ),
        (
            {
                "description": "Skill",
                "sub-skills": ["child", " child "],
            },
            ValueError,
            "field 'sub-skills' cannot contain duplicate values",
        ),
    ],
)
def test_rejects_invalid_metadata(
    data: dict[str, Any],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        DeepSkillMetadataParser().parse(data)


def test_rejects_hds_only_exit_skill() -> None:
    with pytest.raises(ValueError, match="Unknown metadata fields: exit-skill"):
        DeepSkillMetadataParser().parse(
            {
                "description": "Skill",
                "exit-skill": "finish",
            }
        )
