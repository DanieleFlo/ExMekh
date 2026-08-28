from pathlib import Path
from typing import cast

import pytest

from qs.core.agents.capability.markdown_capability import MarkdownCapability
from qs.core.agents.capability.metadata import CapabilityMetadata


class ConcreteMarkdownCapability(MarkdownCapability[CapabilityMetadata]):
    def _render_body(self) -> str:
        return self.body


def test_rejects_non_capability_metadata() -> None:
    with pytest.raises(TypeError, match="metadata must be CapabilityMetadata"):
        ConcreteMarkdownCapability(
            name="capability",
            markdown_path=Path("capability.md"),
            metadata=cast(CapabilityMetadata, object()),
            body="Body",
            references=(),
        )
