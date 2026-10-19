from pathlib import Path
from typing import cast

import pytest

from exmekh_lib.core.agents.capability.markdown_capability import MarkdownCapability
from exmekh_lib.core.agents.capability.metadata import CapabilityMetadata


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


def test_optional_resolution_can_be_repeated_without_changing_declarations():
    metadata = CapabilityMetadata(
        "Description", (), ("required",), optional_sub_tools=("optional",)
    )
    resource = ConcreteMarkdownCapability(
        name="capability",
        markdown_path=Path("capability.md"),
        metadata=metadata,
        body="Body",
        references=(),
    )
    resource.resolve_optional_capabilities({"required", "optional"}, set())
    assert resource.get_sub_tool_names() == ("required", "optional")
    assert resource.copy().get_sub_tool_names() == ("required", "optional")

    resource.resolve_optional_capabilities({"required"}, set())
    assert resource.get_sub_tool_names() == ("required",)
    assert resource.get_direct_tool_names() == {"required"}
    assert metadata.get_sub_tool_names() == ("required",)


def test_direct_metadata_construction_rejects_required_optional_overlap():
    with pytest.raises(ValueError, match="both required and optional: repeated"):
        CapabilityMetadata(
            "Description", (), ("repeated",), optional_sub_tools=("repeated",)
        )
