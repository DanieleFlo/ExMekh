from __future__ import annotations

from pathlib import Path

from exmekh_lib.config.agent_text import AGENT_TEXT
from exmekh_lib.config.capabilities import CAPABILITY_FORMAT
from exmekh_lib.core.agents.capabilities.resource_type import ResourceType
from exmekh_lib.core.agents.capability.markdown_capability import MarkdownCapability
from exmekh_lib.core.agents.hds.metadata import HDSMetadata


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
            raise TypeError(AGENT_TEXT.metadata_must_be_hdsmetadata)

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

    def get_exit_tool_name(self) -> str:
        return self.metadata.get_exit_tool_name()

    def get_agent_context_id(self) -> str:
        agent_context_id = self.metadata.get_agent_context_id()
        if agent_context_id is not None:
            return agent_context_id
        return self.get_name()

    @property
    def agent_context_id(self) -> str:
        return self.get_agent_context_id()

    def __str__(self) -> str:
        return AGENT_TEXT.name_type_hds_body_description.format(
            name=self.name,
            body=self.body[: CAPABILITY_FORMAT.representation_preview_chars],
            description=self.description[
                : CAPABILITY_FORMAT.representation_preview_chars
            ],
        )
