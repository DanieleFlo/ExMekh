from pathlib import Path
from typing import cast

import pytest

from exmekh_lib.core.agents.hds.hds import HDS
from exmekh_lib.core.agents.hds.metadata import HDSMetadata


def _metadata() -> HDSMetadata:
    return HDSMetadata(
        description="Description",
        sub_skills=(),
        sub_tools=(),
        exit_tool="finish",
    )


def test_returns_only_body() -> None:
    hds = HDS(
        name="instructions",
        markdown_path=Path("instructions.md"),
        metadata=_metadata(),
        body="Body",
        references=("search",),
    )

    assert hds.get_hds() == "Body"
    assert hds.tool() == "Body"


def test_exposes_node_configuration() -> None:
    hds = HDS(
        name="instructions",
        markdown_path=Path("instructions.md"),
        metadata=HDSMetadata(
            description="Description",
            sub_skills=("child",),
            sub_tools=("search",),
            exit_tool="finish",
        ),
        body="Body",
        references=(),
    )

    assert hds.get_direct_tool_names() == {"child", "search"}
    assert hds.get_exit_tool_name() == "finish"


def test_rejects_non_hds_metadata() -> None:
    with pytest.raises(TypeError, match="metadata must be HDSMetadata"):
        HDS(
            name="instructions",
            markdown_path=Path("instructions.md"),
            metadata=cast(HDSMetadata, object()),
            body="Body",
            references=(),
        )


def test_returns_configured_agent_context_id() -> None:
    hds = HDS(
        name="instructions",
        markdown_path=Path("instructions.md"),
        metadata=HDSMetadata(
            description="Description",
            sub_skills=(),
            sub_tools=(),
            exit_tool="finish",
            agent_context_id="custom-ctx",
        ),
        body="Body",
        references=(),
    )

    assert hds.get_agent_context_id() == "custom-ctx"
    assert hds.agent_context_id == "custom-ctx"


def test_falls_back_to_name_when_agent_context_id_is_none() -> None:
    hds = HDS(
        name="my-hds-name",
        markdown_path=Path("instructions.md"),
        metadata=_metadata(),
        body="Body",
        references=(),
    )

    assert hds.get_agent_context_id() == "my-hds-name"
    assert hds.agent_context_id == "my-hds-name"
