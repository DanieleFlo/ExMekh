from __future__ import annotations

from pathlib import Path

from qs.core.agents.capabilities.resource_type import ResourceType
from qs.core.agents.capability.markdown_capability import MarkdownCapability
from qs.core.agents.hds.metadata import HDSMetadata


class HDS(MarkdownCapability[HDSMetadata]):
    resource_type = ResourceType.HDS

    def __init__(
        self,
        *,
        name: str,
        markdown_path: Path,
        metadata: HDSMetadata,
        body: str,
        references: tuple[str, ...],
    ) -> None:
        if not isinstance(metadata, HDSMetadata):
            raise TypeError("metadata must be HDSMetadata")

        super().__init__(
            name=name,
            markdown_path=markdown_path,
            metadata=metadata,
            body=body,
            references=references,
        )

    def _render_body(self) -> str:
        return self.body

    def get_hds(self) -> str:
        return self._render_body()
