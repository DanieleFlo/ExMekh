import pytest

from qs.core.agents.hds.metadata import HDSMetadata, HDSMetadataParser


def test_parses_hds_metadata_with_exit_skill() -> None:
    metadata = HDSMetadataParser().parse(
        {
            "description": " Instructions ",
            "sub-skills": [" child "],
            "sub-tools": " search ",
            "exit-skill": " finish ",
        }
    )

    assert metadata == HDSMetadata(
        description="Instructions",
        sub_skills=("child",),
        sub_tools=("search",),
        exit_skill="finish",
    )


def test_gets_direct_tool_names() -> None:
    metadata = HDSMetadata(
        description="Instructions",
        sub_skills=("child",),
        sub_tools=("search",),
        exit_skill="finish",
    )

    assert metadata.get_direct_tool_names() == {"child", "search", "finish"}
    assert metadata.get_exit_skill_name() == "finish"


def test_requires_exit_skill() -> None:
    with pytest.raises(
        ValueError, match="Missing required metadata fields: exit-skill"
    ):
        HDSMetadataParser().parse({"description": "Instructions"})


def test_rejects_deep_skill_only_metadata_field() -> None:
    with pytest.raises(
        ValueError,
        match="Unknown metadata fields: inherits-parent-tools",
    ):
        HDSMetadataParser().parse(
            {
                "description": "Instructions",
                "exit-skill": "finish",
                "inherits-parent-tools": True,
            }
        )
