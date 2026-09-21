from __future__ import annotations

import re
from typing import Any, Final, Protocol

import yaml

from qs.core.agents.capability.markdown import CapabilityMarkdown
from qs.core.agents.capability.metadata import (
    CapabilityMetadata,
)
from qs.core.agents.capability.references import parse_references


class MetadataParser[MetadataType: CapabilityMetadata](Protocol):
    def parse(self, data: dict[str, Any]) -> MetadataType: ...


class CapabilityMarkdownParser[
    MetadataType: CapabilityMetadata,
    MarkdownType: CapabilityMarkdown,
]:
    FRONT_MATTER_PATTERN: Final[re.Pattern[str]] = re.compile(
        r"\A"
        r"---[ \t]*\r?\n"
        r"(?P<head>.*?)"
        r"\r?\n---[ \t]*"
        r"(?:\r?\n|\Z)"
        r"(?P<body>.*)"
        r"\Z",
        re.DOTALL,
    )

    def __init__(
        self,
        metadata_parser: MetadataParser[MetadataType],
        markdown_type: type[MarkdownType],
    ) -> None:
        self._metadata_parser = metadata_parser
        self._markdown_type = markdown_type

    def parse(self, markdown: str) -> MarkdownType:
        if not isinstance(markdown, str):
            raise TypeError("markdown must be a string")

        head, body = self._split_markdown(markdown)
        metadata = self._parse_metadata(head)
        parsed_body, references = self._parse_body(body)

        return self._markdown_type(
            metadata=metadata,
            body=parsed_body,
            references=references,
        )

    def _split_markdown(self, markdown: str) -> tuple[str, str]:
        match = self.FRONT_MATTER_PATTERN.match(markdown)

        if match is None:
            raise ValueError(
                "Markdown must contain a valid YAML front matter "
                "section delimited by '---'."
            )

        return match.group("head"), match.group("body")

    def _parse_metadata(self, head: str) -> MetadataType:
        return self._metadata_parser.parse(self._load_yaml(head))

    @staticmethod
    def _load_yaml(head: str) -> dict[str, Any]:
        try:
            data = yaml.safe_load(head)
        except yaml.YAMLError as error:
            raise ValueError(f"Invalid YAML metadata: {error}") from error

        if data is None:
            raise ValueError("Metadata section cannot be empty.")

        if not isinstance(data, dict):
            raise TypeError("Metadata section must be a YAML mapping.")

        return data

    def _parse_body(self, body: str) -> tuple[str, tuple[str, ...]]:
        return parse_references(body)
