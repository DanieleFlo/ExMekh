from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal
from xml.etree import ElementTree as ET

from qs.config.runtime import RUNTIME
from qs.config.validation import VALIDATION_PATTERNS
from qs.utils.json import to_json_value


@dataclass
class XmlHeader:
    source: Literal["Tool", "DeepSkill"]
    name: str
    id: str
    fields: dict[str, Any] = field(default_factory=dict)
    args: dict[str, Any] | None = None

    ROOT_TAG = RUNTIME.xml_header_tag
    _XML_NAME = re.compile(VALIDATION_PATTERNS.xml_name)

    def __post_init__(self) -> None:
        if not isinstance(self.source, str):
            raise TypeError("source must be a string")
        if self.source not in ("Tool", "DeepSkill"):
            raise ValueError(
                f"source must be 'Tool' or 'DeepSkill', got '{self.source}'"
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

        ET.SubElement(root, "name").text = self.name
        ET.SubElement(root, "type").text = self.source
        ET.SubElement(root, "id").text = self.id

        if self.args is not None:
            args_element = ET.SubElement(root, "args")
            for name, value in self.args.items():
                if self._XML_NAME.fullmatch(name):
                    argument = ET.SubElement(args_element, name)
                else:
                    argument = ET.SubElement(args_element, "arg", {"name": name})
                argument.text = self._format_argument(value)

        for key, value in self.fields.items():
            if value is None:
                continue

            ET.SubElement(root, key).text = str(value)

        ET.indent(root)

        return ET.tostring(
            root,
            encoding="unicode",
            short_empty_elements=True,
        )

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
