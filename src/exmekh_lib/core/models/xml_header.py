from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal
from xml.etree import ElementTree as ET

from exmekh_lib.config.runtime import RUNTIME
from exmekh_lib.config.validation import VALIDATION_PATTERNS
from exmekh_lib.utils.json import to_json_value
from exmekh_lib.utils.xml import append_text_element, append_value_element, render_xml


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
            raise TypeError("source must be a string")
        if self.source not in ("Tool", "DeepSkill", "UserMessage"):
            raise ValueError(
                f"source must be 'Tool', 'DeepSkill', or 'UserMessage', got '{self.source}'"
            )

        if not isinstance(self.name, str):
            raise TypeError("name must be a string")
        name = self.name.strip()
        if not name:
            raise ValueError("name cannot be empty")
        self.name = name

        if not isinstance(self.id, str):
            raise TypeError("id must be a string")
        header_id = self.id.strip()
        if not header_id:
            raise ValueError("id cannot be empty")
        self.id = header_id

        if not isinstance(self.fields, dict):
            raise TypeError("fields must be a dictionary")
        if not all(isinstance(key, str) for key in self.fields):
            raise TypeError("all keys in fields must be strings")

        if self.args is not None and not isinstance(self.args, dict):
            raise TypeError("args must be a dictionary or None")
        if self.args is not None and not all(isinstance(key, str) for key in self.args):
            raise TypeError("all keys in args must be strings")

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

        for key, value in self.fields.items():
            if value is None:
                continue

            if isinstance(value, (dict, list)):
                append_value_element(root, key, value)
            else:
                append_text_element(root, key, str(value))

        return render_xml(root)

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
    def body_from_content(cls, content: str) -> str | None:
        """Read the body only when content starts with a valid XML header."""
        leading = cls._leading_header(content)
        if leading is None:
            return None
        return content[cls._body_start(content, leading[1]) :]

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
