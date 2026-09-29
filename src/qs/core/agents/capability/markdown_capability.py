from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Self

from qs.core.agents.capabilities.resource_type import ResourceType
from qs.core.agents.capability.loader import MarkdownLoader
from qs.core.agents.capability.markdown import CapabilityMarkdown
from qs.core.agents.capability.metadata import CapabilityMetadata
from qs.core.agents.capability.parser import CapabilityMarkdownParser
from qs.core.models.tool import Tool


class MarkdownCapability[MetadataType: CapabilityMetadata](Tool, ABC):
    resource_type = ResourceType.MARKDOWN_CAPABILITY

    def __init__(
        self,
        *,
        name: str,
        markdown_path: Path,
        metadata: MetadataType,
        body: str,
        references: tuple[str, ...],
    ) -> None:
        self._validate_inputs(markdown_path, metadata, body, references)
        self.markdown_path = markdown_path
        self.metadata = metadata
        self.body = body
        self.references = references
        super().__init__(
            tool=self._render_body,
            name=name,
            description=metadata.get_description(),
            background_enabled=False,
        )

    @abstractmethod
    def _render_body(self) -> str:
        pass  # pragma: no cover

    def get_sub_skill_names(self) -> tuple[str, ...]:
        return self.metadata.get_sub_skill_names()

    def get_sub_tool_names(self) -> tuple[str, ...]:
        return self.metadata.get_sub_tool_names()

    def get_direct_tool_names(self) -> set[str]:
        return self.metadata.get_direct_tool_names()

    @classmethod
    def from_markdown[MarkdownType: CapabilityMarkdown[Any]](
        cls,
        markdown_path: Path,
        parser: CapabilityMarkdownParser[MetadataType, MarkdownType],
    ) -> Self:
        markdown = parser.parse(MarkdownLoader(markdown_path).load())
        return cls(
            name=markdown_path.stem,
            markdown_path=markdown_path,
            metadata=markdown.metadata,
            body=markdown.body,
            references=markdown.references,
        )

    @staticmethod
    def _validate_inputs(
        markdown_path: Path,
        metadata: MetadataType,
        body: str,
        references: tuple[str, ...],
    ) -> None:
        if not isinstance(markdown_path, Path):
            raise TypeError("markdown_path must be a Path")

        if not isinstance(metadata, CapabilityMetadata):
            raise TypeError("metadata must be CapabilityMetadata")

        if not isinstance(body, str):
            raise TypeError("body must be a string")

        if not isinstance(references, tuple):
            raise TypeError("references must be a tuple")

        if not all(isinstance(reference, str) for reference in references):
            raise TypeError("references must contain only strings")

        if len(references) != len(set(references)):
            raise ValueError("references cannot contain duplicates")
