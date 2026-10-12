"""Restricted XML edit grammar, without XML newline normalization or filesystem access."""

import re
from dataclasses import dataclass
from typing import Literal, Self

from exmekh_lib.config.file_editing import FILE_EDITING

_SPACE = " \t\r\n"
_ENTITY = re.compile(r"&(?:amp|lt|gt|quot|apos|#[0-9]+|#x[0-9a-fA-F]+);")
_ENTITIES = {"amp": "&", "lt": "<", "gt": ">", "quot": '"', "apos": "'"}


def _valid_character(character: str) -> bool:
    value = ord(character)
    return (
        value in (9, 10, 13)
        or 0x20 <= value <= 0xD7FF
        or 0xE000 <= value <= 0xFFFD
        or 0x10000 <= value <= 0x10FFFF
    )


def _decode_entity(entity: str) -> str:
    name = entity[1:-1]
    if name in _ENTITIES:
        return _ENTITIES[name]
    digits = (name[2:] if name.startswith("#x") else name[1:]).lstrip("0") or "0"
    if len(digits) > 8:
        raise ValueError("XML character reference is outside the Unicode range")
    try:
        character = chr(int(digits, 16 if name.startswith("#x") else 10))
    except (ValueError, OverflowError) as error:
        raise ValueError("Invalid XML character reference") from error
    if not _valid_character(character):
        raise ValueError("Invalid XML character reference")
    return character


@dataclass(frozen=True, slots=True)
class EditSelection:
    kind: Literal["literal", "all", "line", "range"]
    text: str = ""
    start_line: str = ""
    end_line: str = ""


@dataclass(frozen=True, slots=True)
class EditElement:
    path: str
    selection: EditSelection
    replacement: str | None


@dataclass(frozen=True, slots=True)
class EditDocument:
    elements: tuple[EditElement, ...]

    @classmethod
    def parse(cls, source: str) -> Self:
        if not isinstance(source, str):
            raise ValueError("The edit document must be an exact string")
        if any(not _valid_character(character) for character in source):
            raise ValueError("The edit document contains invalid XML characters")
        if len(source.encode("utf-8")) > FILE_EDITING.max_generated_bytes:
            raise ValueError("Edit document exceeds the generated UTF-8 byte limit")
        parser = _EditParser(source)
        elements: list[EditElement] = []
        parser.skip_space()
        while not parser.finished:
            if len(elements) >= FILE_EDITING.max_exchanges:
                raise ValueError("Edit document exceeds the element limit")
            elements.append(parser.element())
            parser.skip_space()
        if not elements:
            raise ValueError("Edit document must contain at least one edit")
        return cls(tuple(elements))


class _EditParser:
    def __init__(self, source: str) -> None:
        self._source = source
        self._position = 0

    @property
    def finished(self) -> bool:
        return self._position == len(self._source)

    def skip_space(self) -> None:
        while not self.finished and self._source[self._position] in _SPACE:
            self._position += 1

    def _take(self, token: str) -> None:
        if not self._source.startswith(token, self._position):
            raise ValueError(
                f"Expected {token!r} at edit document offset {self._position}"
            )
        self._position += len(token)

    def _match(self, pattern: str) -> re.Match[str] | None:
        match = re.compile(pattern).match(self._source, self._position)
        if match is not None:
            self._position = match.end()
        return match

    def _text(self, closing: str, *, replacement: bool = False) -> str:
        pieces: list[str] = []
        while not self.finished:
            if self._source.startswith(closing, self._position):
                tail = self._position + len(closing)
                while tail < len(self._source) and self._source[tail] in _SPACE:
                    tail += 1
                if not replacement or self._source.startswith("</edit>", tail):
                    self._take(closing)
                    return "".join(pieces)
                pieces.append(closing)
                self._position += len(closing)
            elif self._source.startswith("<![CDATA[", self._position):
                start = self._position + 9
                end = self._source.find("]]>", start)
                if end < 0:
                    raise ValueError("Unterminated CDATA in edit document")
                pieces.append(self._source[start:end])
                self._position = end + 3
            elif self._source[self._position] == "&":
                entity = _ENTITY.match(self._source, self._position)
                if entity is None:
                    raise ValueError("Unknown or unterminated XML entity")
                pieces.append(_decode_entity(entity[0]))
                self._position = entity.end()
            elif self._source[self._position] == "<" or self._source.startswith(
                "]]>", self._position
            ):
                raise ValueError("Literal XML markup requires entities or CDATA")
            else:
                pieces.append(self._source[self._position])
                self._position += 1
        raise ValueError(f"Missing {closing} in edit document")

    def element(self) -> EditElement:
        self._take('<edit path="')
        end = self._source.find('"', self._position)
        if end < 0:
            raise ValueError("Unterminated edit path")
        raw_path = self._source[self._position : end]
        if not raw_path or "<" in raw_path:
            raise ValueError("Edit path must be a nonempty XML attribute")
        path = self._text('"')
        self._take(">")
        range_match = self._match(
            r'<old_text start_line="([0-9]+)" end_line="([0-9]+)"></old_text>'
        )
        if range_match is not None:
            selection = EditSelection(
                "range", start_line=range_match[1], end_line=range_match[2]
            )
        else:
            self._take("<old_text>")
            if self._match(r"<all/></old_text>") is not None:
                selection = EditSelection("all")
            elif (
                line := self._match(r'<line number="([0-9]+)"/></old_text>')
            ) is not None:
                selection = EditSelection("line", start_line=line[1])
            else:
                selection = EditSelection("literal", self._text("</old_text>"))
        self._take("<new_text>")
        if self._source.startswith("<delete/>", self._position):
            self._take("<delete/></new_text>")
            replacement = None
        else:
            replacement = self._text("</new_text>", replacement=True)
        self.skip_space()
        self._take("</edit>")
        return EditElement(path, selection, replacement)
