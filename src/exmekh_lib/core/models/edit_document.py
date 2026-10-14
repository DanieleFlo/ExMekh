"""Two-marker edit framing; file bodies are always exact, literal text."""

import re
from dataclasses import dataclass
from html import unescape
from typing import Self

from exmekh_lib.config.file_editing import FILE_EDIT_FORMAT, FILE_EDITING
from exmekh_lib.utils.tool_errors import ToolInputError

_OPEN = re.compile(FILE_EDIT_FORMAT.opening_pattern)
_SEPARATOR = re.compile(FILE_EDIT_FORMAT.separator_pattern)


def literal_selection_offset(text: str, old: str) -> int | None:
    """Locate a globally unique literal beginning at a physical line boundary."""
    if not text:
        return 0 if not old else None
    if not old:
        return None
    offset = text.find(old)
    if offset < 0 or text.find(old, offset + 1) >= 0:
        return None
    if (
        offset
        and text[offset - 1] != "\n"
        and not (text[offset - 1] == "\r" and text[offset] != "\n")
    ):
        return None
    return offset


@dataclass(frozen=True, slots=True)
class EditElement:
    path: str
    old_text: str
    replacement: str
    identifier: str | None = None


@dataclass(frozen=True, slots=True)
class EditDocument:
    elements: tuple[EditElement, ...]

    @classmethod
    def parse(cls, source: str, *, snapshots: tuple[str, ...] = ()) -> Self:
        cls._validate_source(source)
        offset = len(source) - len(source.lstrip(" \t\r\n"))
        elements: list[EditElement] = []
        used_ids: set[str] = set()
        while offset < len(source):
            opening = _OPEN.match(source, offset)
            if opening is None:
                raise ToolInputError(
                    FILE_EDITING.opening_marker_required, code="invalid_edit_document"
                )
            identifier = opening[2]
            cls._reserve_identifier(identifier, used_ids)
            boundary = next(
                (
                    match
                    for match in _OPEN.finditer(source, opening.end())
                    if match[2] is None or match[2] not in used_ids
                ),
                None,
            )
            end = boundary.start() if boundary else len(source)
            snapshot = (
                snapshots[len(elements)] if len(elements) < len(snapshots) else None
            )
            separator = cls._select_separator(source, opening, end, snapshot)
            if identifier is None and separator[1] is not None:
                identifier = separator[1]
                cls._reserve_identifier(identifier, used_ids)
            elements.append(
                EditElement(
                    unescape(opening[1]),
                    source[opening.end() : separator.start()],
                    source[separator.end() : end],
                    identifier,
                )
            )
            if len(elements) > FILE_EDITING.max_exchanges:
                raise ToolInputError(
                    FILE_EDITING.document_element_limit, code="invalid_edit_document"
                )
            offset = end
        if not elements:
            raise ToolInputError(
                FILE_EDITING.empty_document, code="invalid_edit_document"
            )
        return cls(tuple(elements))

    @staticmethod
    def _validate_source(source: str) -> None:
        if not isinstance(source, str):
            raise ToolInputError(
                FILE_EDITING.document_string, code="invalid_edit_document"
            )
        if "\x00" in source or any(0xD800 <= ord(ch) <= 0xDFFF for ch in source):
            raise ToolInputError(
                FILE_EDITING.document_unicode, code="invalid_edit_document"
            )
        if len(source.encode("utf-8")) > FILE_EDITING.max_generated_bytes:
            raise ToolInputError(
                FILE_EDITING.document_byte_limit, code="file_limit_exceeded"
            )

    @staticmethod
    def _reserve_identifier(identifier: str | None, used_ids: set[str]) -> None:
        if identifier is None:
            return
        if identifier in used_ids:
            raise ToolInputError(
                FILE_EDITING.duplicate_edit_id, code="invalid_edit_document"
            )
        used_ids.add(identifier)

    @staticmethod
    def _select_separator(
        source: str, opening: re.Match[str], end: int, snapshot: str | None
    ) -> re.Match[str]:
        identifier = opening[2]
        separators = [
            match
            for match in _SEPARATOR.finditer(source, opening.end(), end)
            if match[1] is None or identifier is None or match[1] == identifier
        ]
        explicit = [
            match
            for match in separators
            if identifier is not None and match[1] == identifier
        ]
        if explicit:
            separators = explicit
        if not separators:
            raise ToolInputError(
                FILE_EDITING.separator_required, code="invalid_edit_document"
            )
        if len(separators) > 1 and snapshot is not None:
            separators = [
                match
                for match in separators
                if literal_selection_offset(
                    snapshot, source[opening.end() : match.start()]
                )
                is not None
            ]
        if len(separators) != 1:
            raise ToolInputError(
                FILE_EDITING.separator_ambiguous, code="invalid_edit_document"
            )
        return separators[0]
