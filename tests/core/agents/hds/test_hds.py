from pathlib import Path
from typing import cast

import pytest

from qs.core.agents.hds.hds import HDS
from qs.core.agents.hds.metadata import HDSMetadata


def _metadata() -> HDSMetadata:
    return HDSMetadata(
        description="Description",
        sub_skills=(),
        sub_tools=(),
        exit_skill="finish",
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


def test_rejects_non_hds_metadata() -> None:
    with pytest.raises(TypeError, match="metadata must be HDSMetadata"):
        HDS(
            name="instructions",
            markdown_path=Path("instructions.md"),
            metadata=cast(HDSMetadata, object()),
            body="Body",
            references=(),
        )
