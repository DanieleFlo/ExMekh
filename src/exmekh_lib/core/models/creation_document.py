"""Creation framing with literal bodies and independently rejected blocks."""

import re
from dataclasses import dataclass
from html import unescape
from typing import Literal, Self

from exmekh_lib.config.file_editing import FILE_CREATION_FORMAT, FILE_EDITING
from exmekh_lib.utils.tool_errors import ToolInputError

_MARKER = re.compile(FILE_CREATION_FORMAT.marker_pattern)
_OPENING = re.compile(FILE_CREATION_FORMAT.opening_pattern)


@dataclass(frozen=True, slots=True)
class CreationElement:
    resource_type: Literal["file", "directory"]
    path: str
    content: str


@dataclass(frozen=True, slots=True)
class CreationDocument:
    elements: tuple[CreationElement | ToolInputError, ...]

    @classmethod
    def parse(cls, source: str) -> Self:
        cls._validate_source(source)
        markers = list(_MARKER.finditer(source))
        if not markers:
            raise ToolInputError(
                FILE_EDITING.empty_creation_document, code="invalid_creation_document"
            )
        if len(markers) > FILE_EDITING.max_creation_elements:
            raise ToolInputError(
                FILE_EDITING.creation_document_element_limit.format(
                    maximum=FILE_EDITING.max_creation_elements
                ),
                code="file_limit_exceeded",
                hint=FILE_EDITING.creation_document_limit_hint,
            )
        elements: list[CreationElement | ToolInputError] = []
        if source[: markers[0].start()].strip():
            elements.append(cls._framing_error())
        for index, marker in enumerate(markers):
            end = (
                markers[index + 1].start() if index + 1 < len(markers) else len(source)
            )
            elements.append(cls._parse_element(source, marker.start(), end))
        return cls(tuple(elements))

    @staticmethod
    def _validate_source(source: str) -> None:
        if not isinstance(source, str):
            raise ToolInputError(
                FILE_EDITING.creation_document_string, code="invalid_creation_document"
            )
        # Invalid Unicode inside a block is rejected independently by its operation.
        if (
            len(source.encode("utf-8", "surrogatepass"))
            > FILE_EDITING.max_creation_document_bytes
        ):
            raise ToolInputError(
                FILE_EDITING.creation_document_byte_limit.format(
                    maximum=FILE_EDITING.max_creation_document_bytes
                ),
                code="file_limit_exceeded",
                hint=FILE_EDITING.creation_document_limit_hint,
            )

    @classmethod
    def _parse_element(
        cls, source: str, start: int, end: int
    ) -> CreationElement | ToolInputError:
        opening = _OPENING.match(source, start, end)
        if opening is None:
            return cls._framing_error()
        return CreationElement(
            "file" if opening[1] == "file" else "directory",
            unescape(opening[2]),
            source[opening.end() : end],
        )

    @staticmethod
    def _framing_error() -> ToolInputError:
        return ToolInputError(
            FILE_EDITING.creation_marker_required, code="invalid_creation_document"
        )
