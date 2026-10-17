from __future__ import annotations

import base64
import re
from copy import deepcopy
from xml.etree import ElementTree as ET

from exmekh_lib.config.validation import VALIDATION_PATTERNS
from exmekh_lib.utils.json import JsonValue, to_json_value

_XML_NAME = re.compile(VALIDATION_PATTERNS.xml_name)
_INVALID_XML_CHARACTER = re.compile(VALIDATION_PATTERNS.xml_invalid_character)


def stringify_tool_result(value: object) -> str:
    """Render structured results as XML while preserving tool-authored text."""
    if isinstance(value, str):
        return value
    root = ET.Element("tool-result")
    _append_value(root, to_json_value(value))
    return render_xml(root)


def render_xml(root: ET.Element) -> str:
    """Serialize an XML tree, preserving carriage returns when it is parsed."""
    formatted = deepcopy(root)
    ET.indent(formatted)
    return ET.tostring(formatted, encoding="unicode").replace("\r", "&#13;")


def read_text_element(element: ET.Element) -> str:
    """Read XML text using the same lossless encoding as the shared renderer."""
    text = element.text or ""
    if element.get("encoding") == "base64":
        return base64.b64decode(text, validate=True).decode("utf-8", "surrogatepass")
    return text


def append_text_element(
    parent: ET.Element,
    name: str,
    text: str,
    *,
    attributes: dict[str, str] | None = None,
) -> ET.Element:
    """Append XML text, encoding characters that XML cannot represent."""
    element = create_text_element(name, text, attributes=attributes)
    parent.append(element)
    return element


def create_text_element(
    name: str,
    text: str,
    *,
    attributes: dict[str, str] | None = None,
) -> ET.Element:
    """Create XML text with the same encoding rules for root and child elements."""
    element = ET.Element(name)
    _set_text(element, text)
    for key, value in (attributes or {}).items():
        if _INVALID_XML_CHARACTER.search(value):
            element.set(f"{key}-encoding", "base64")
            value = _base64_text(value)
        element.set(key, value)
    return element


def append_value_element(parent: ET.Element, name: str, value: object) -> ET.Element:
    """Append a structured value with the same types and escaping as tool results."""
    element = ET.SubElement(parent, name)
    _append_value(element, to_json_value(value))
    return element


def _base64_text(text: str) -> str:
    return base64.b64encode(text.encode("utf-8", "surrogatepass")).decode("ascii")


def _set_text(element: ET.Element, text: str) -> None:
    if _INVALID_XML_CHARACTER.search(text):
        element.set("encoding", "base64")
        element.text = _base64_text(text)
    else:
        element.text = text


def _append_value(element: ET.Element, value: JsonValue) -> None:
    if isinstance(value, dict):
        element.set("type", "object")
        _append_object_fields(element, value)
    elif isinstance(value, list):
        element.set("type", "array")
        for item in value:
            _append_value(ET.SubElement(element, "item"), item)
    elif value is None:
        element.set("type", "null")
    elif isinstance(value, bool):
        element.set("type", "boolean")
        element.text = "true" if value else "false"
    elif isinstance(value, str):
        element.set("type", "string")
        _set_text(element, value)
    else:
        element.set("type", "integer" if isinstance(value, int) else "number")
        element.text = str(value)


def _append_object_fields(element: ET.Element, value: dict[str, JsonValue]) -> None:
    for key, item in value.items():
        if _XML_NAME.fullmatch(key):
            child = ET.SubElement(element, key)
        else:
            field = ET.SubElement(element, "field")
            append_text_element(field, "key", key)
            child = ET.SubElement(field, "value")
        _append_value(child, item)
