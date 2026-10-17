from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal
from xml.etree import ElementTree as ET

from exmekh_lib.config.model_text import MODEL_TEXT
from exmekh_lib.config.runtime import RUNTIME
from exmekh_lib.config.validation import VALIDATION_PATTERNS
from exmekh_lib.core.models.tool_output_reference import ToolOutputReference
from exmekh_lib.utils.json import to_json_value
from exmekh_lib.utils.time import format_timestamp, format_timing_fields
from exmekh_lib.utils.xml import (
    append_text_element,
    append_value_element,
    create_text_element,
    read_text_element,
    render_xml,
)


@dataclass(frozen=True, slots=True)
class XmlHeaderContent:
    """Read parsed metadata without exposing its XML representation."""

    header: str
    body: str
    _root: ET.Element = field(repr=False, compare=False)

    def field(self, key: str) -> str | None:
        element = self._root.find(key)
        return None if element is None else read_text_element(element)

    def output_references(self) -> tuple[ToolOutputReference, ...] | None:
        container = self._root.find("used-tools")
        if container is None:
            return None
        return tuple(
            ToolOutputReference.from_dict(
                {child.tag: read_text_element(child) for child in item}
            )
            for item in container.findall("item")
        )


@dataclass
class XmlHeader:
    source: Literal["Tool", "DeepSkill", "UserMessage"]
    name: str
    id: str
    fields: dict[str, Any] = field(default_factory=dict)
    args: dict[str, Any] | None = None

    ROOT_TAG = RUNTIME.xml_header_tag
    _XML_NAME = re.compile(VALIDATION_PATTERNS.xml_name)

    def __post_init__(self) -> None:
        if not isinstance(self.source, str):
            raise TypeError(MODEL_TEXT.source_must_be_a_string)
        if self.source not in ("Tool", "DeepSkill", "UserMessage"):
            raise ValueError(
                MODEL_TEXT.source_must_be_tool_deepskill_or_usermessage_got.format(
                    source=self.source
                )
            )

        if not isinstance(self.name, str):
            raise TypeError(MODEL_TEXT.name_must_be_a_string)
        name = self.name.strip()
        if not name:
            raise ValueError(MODEL_TEXT.name_cannot_be_empty)
        self.name = name

        if not isinstance(self.id, str):
            raise TypeError(MODEL_TEXT.id_must_be_a_string)
        header_id = self.id.strip()
        if not header_id:
            raise ValueError(MODEL_TEXT.id_cannot_be_empty)
        self.id = header_id

        if not isinstance(self.fields, dict):
            raise TypeError(MODEL_TEXT.fields_must_be_a_dictionary)
        if not all(isinstance(key, str) for key in self.fields):
            raise TypeError(MODEL_TEXT.all_keys_in_fields_must_be_strings)

        if self.args is not None and not isinstance(self.args, dict):
            raise TypeError(MODEL_TEXT.args_must_be_a_dictionary_or_none)
        if self.args is not None and not all(isinstance(key, str) for key in self.args):
            raise TypeError(MODEL_TEXT.all_keys_in_args_must_be_strings)

    def add_field(self, key: str, value: Any) -> XmlHeader:
        """Add or replace a custom metadata field."""
        self.fields[key] = value
        return self

    def add_fields(self, **fields: Any) -> XmlHeader:
        """Add multiple custom metadata fields."""
        self.fields.update(fields)
        return self

    def remove_field(self, key: str) -> XmlHeader:
        """Remove a custom field if present."""
        self.fields.pop(key, None)
        return self

    def to_xml(self) -> str:
        root = ET.Element(self.ROOT_TAG)

        append_text_element(root, "name", self.name)
        append_text_element(root, "type", self.source)
        append_text_element(root, "id", self.id)
        root.append(self._current_time_element())

        if self.args is not None:
            args_element = ET.SubElement(root, "args")
            for name, value in self.args.items():
                if self._XML_NAME.fullmatch(name):
                    append_text_element(
                        args_element, name, self._format_argument(value)
                    )
                else:
                    append_text_element(
                        args_element,
                        "arg",
                        self._format_argument(value),
                        attributes={"name": name},
                    )

        for key, value in format_timing_fields(self.fields).items():
            if value is None or key == "current-time":
                continue

            if isinstance(value, (dict, list)):
                append_value_element(root, key, value)
            else:
                append_text_element(root, key, str(value))

        return render_xml(root)

    @classmethod
    def refresh_timing_in_content(cls, content: str) -> str:
        """Refresh response time and legacy timing fields, preserving the body."""
        leading = cls._leading_header(content)
        if leading is None:
            return content
        root, end = leading
        for existing in root.findall("current-time"):
            root.remove(existing)
        identity = root.find("id")
        position = len(root) if identity is None else list(root).index(identity) + 1
        root.insert(position, cls._current_time_element())
        cls._refresh_legacy_timing(root)
        return render_xml(root) + content[end:]

    @staticmethod
    def _refresh_legacy_timing(root: ET.Element) -> None:
        for position, element in enumerate(list(root)):
            if element.tag not in ("started-at", "finished-at", "elapsed-seconds"):
                continue
            try:
                text = read_text_element(element)
            except (ValueError, UnicodeError):
                # Unknown legacy encodings must not prevent reading the saved body.
                continue
            fields = format_timing_fields({element.tag: text})
            for name, value in fields.items():
                if name == element.tag and value == text:
                    continue
                attributes = dict(element.attrib)
                attributes.pop("encoding", None)
                if "type" in attributes:
                    attributes["type"] = "string"
                replacement = create_text_element(
                    name, str(value), attributes=attributes
                )
                root.remove(element)
                root.insert(position, replacement)

    @staticmethod
    def _current_time_element() -> ET.Element:
        return create_text_element("current-time", format_timestamp(datetime.now(UTC)))

    @staticmethod
    def _format_argument(value: Any) -> str:
        if isinstance(value, str):
            return value
        return json.dumps(
            to_json_value(value),
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def prepend_to(self, content: str) -> str:
        """Prepend the XML metadata block to the given content."""
        return f"{self.to_xml()}\n{content}"

    @classmethod
    def _leading_header(cls, content: str) -> tuple[ET.Element, int] | None:
        closing = f"</{cls.ROOT_TAG}>"
        if not content.startswith(f"<{cls.ROOT_TAG}>"):
            return None
        end = content.find(closing)
        if end < 0:
            return None
        end += len(closing)
        try:
            root = ET.fromstring(content[:end])
        except ET.ParseError:
            return None
        return root, end

    @classmethod
    def has_leading_header(cls, content: str) -> bool:
        return cls._leading_header(content) is not None

    @classmethod
    def parse_content(cls, content: str) -> XmlHeaderContent | None:
        """Parse once, preserving the body after exactly one framing separator."""
        leading = cls._leading_header(content)
        if leading is None:
            return None
        root, end = leading
        return XmlHeaderContent(
            content[:end], content[cls._body_start(content, end) :], root
        )

    @classmethod
    def field_from_content(cls, content: str, key: str) -> str | None:
        parsed = cls.parse_content(content)
        return None if parsed is None else parsed.field(key)

    @classmethod
    def output_references_from_content(
        cls, content: str
    ) -> tuple[ToolOutputReference, ...]:
        parsed = cls.parse_content(content)
        return () if parsed is None else parsed.output_references() or ()

    @classmethod
    def body_from_content(cls, content: str) -> str | None:
        """Read the body only when content starts with a valid XML header."""
        parsed = cls.parse_content(content)
        return None if parsed is None else parsed.body

    @classmethod
    def replace_body_in_content(cls, content: str, body: str) -> str:
        """Replace the body while preserving the header and its line separator."""
        leading = cls._leading_header(content)
        if leading is None:
            return content
        return content[: cls._body_start(content, leading[1])] + body

    @staticmethod
    def _body_start(content: str, header_end: int) -> int:
        if content.startswith("\r\n", header_end):
            return header_end + 2
        if content.startswith("\n", header_end):
            return header_end + 1
        return header_end

    @classmethod
    def add_field_to_content(cls, content: str, key: str, value: str) -> str:
        """Add or replace metadata in a valid header, preserving the body."""
        leading = cls._leading_header(content)
        if leading is None:
            return content
        root, end = leading
        for existing in root.findall(key):
            root.remove(existing)
        append_text_element(root, key, value)
        return render_xml(root) + content[end:]
