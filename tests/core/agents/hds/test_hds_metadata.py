import pytest

from qs.core.agents.hds.metadata import HDSMetadata, HDSMetadataParser


def test_parses_hds_metadata_with_exit_tool() -> None:
    metadata = HDSMetadataParser().parse(
        {
            "description": " Instructions ",
            "sub-skills": [" child "],
            "sub-tools": " search ",
            "exit-tool": " finish ",
        }
    )

    assert metadata == HDSMetadata(
        description="Instructions",
        sub_skills=("child",),
        sub_tools=("search",),
        exit_tool="finish",
    )


def test_gets_direct_tool_names() -> None:
    metadata = HDSMetadata(
        description="Instructions",
        sub_skills=("child",),
        sub_tools=("search",),
        exit_tool="finish",
    )

    assert metadata.get_direct_tool_names() == {"child", "search"}
    assert metadata.get_exit_tool_name() == "finish"


def test_requires_exit_tool() -> None:
    with pytest.raises(ValueError, match="Missing required metadata fields: exit-tool"):
        HDSMetadataParser().parse({"description": "Instructions"})


def test_rejects_deep_skill_only_metadata_field() -> None:
    with pytest.raises(
        ValueError,
        match="Unknown metadata fields: inherits-parent-tools",
    ):
        HDSMetadataParser().parse(
            {
                "description": "Instructions",
                "exit-tool": "finish",
                "inherits-parent-tools": True,
            }
        )


@pytest.mark.parametrize("value", [None, [], ["first", "second"], 3, "", " "])
def test_requires_exactly_one_nonblank_exit_tool_name(value):
    with pytest.raises((TypeError, ValueError), match="exit-tool"):
        HDSMetadataParser().parse({"description": "Instructions", "exit-tool": value})


def test_rejects_legacy_exit_skill_metadata():
    with pytest.raises(ValueError, match="Unknown metadata fields: exit-skill"):
        HDSMetadataParser().parse(
            {"description": "Instructions", "exit-skill": "finish"}
        )


def test_parses_hds_metadata_with_agent_context_id() -> None:
    metadata = HDSMetadataParser().parse(
        {
            "description": "Instructions",
            "exit-tool": "finish",
            "agent-context-id": " custom-agent-ctx ",
        }
    )

    assert metadata.get_agent_context_id() == "custom-agent-ctx"
    assert metadata.agent_context_id == "custom-agent-ctx"


def test_parses_hds_metadata_without_agent_context_id_defaults_to_none() -> None:
    metadata = HDSMetadataParser().parse(
        {
            "description": "Instructions",
            "exit-tool": "finish",
        }
    )

    assert metadata.get_agent_context_id() is None
    assert metadata.agent_context_id is None


@pytest.mark.parametrize("value", [[], ["first"], 3, "", "   "])
def test_rejects_invalid_agent_context_id(value: object) -> None:
    with pytest.raises((TypeError, ValueError), match="agent-context-id"):
        HDSMetadataParser().parse(
            {
                "description": "Instructions",
                "exit-tool": "finish",
                "agent-context-id": value,
            }
        )
