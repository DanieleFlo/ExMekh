"""Decode model-facing structured XML in integration tests."""

import base64
import binascii
import math
import re
from xml.etree import ElementTree as ET

from qs.utils.json import JsonValue


def parse_tool_result(content: str) -> JsonValue:
    if not isinstance(content, str):
        raise TypeError("Tool result content must be a string")
    body = content.split("</header>", 1)[-1].strip()
    return _value(ET.fromstring(body))


def parse_used_tools(content: str) -> JsonValue:
    """Decode skill output references using the shared typed XML parser."""
    header = content.split("</header>", 1)[0] + "</header>"
    return _value(ET.fromstring(header).find("used-tools"))


def _value(element: ET.Element | None) -> JsonValue:
    if element is None:
        raise ValueError("Missing XML value element")
    if not isinstance(element, ET.Element):
        raise TypeError("XML value must be an Element")
    kind = element.get("type")
    encoding = element.get("encoding")
    if encoding is not None and (encoding != "base64" or kind not in (None, "string")):
        raise ValueError("Only string values support base64 encoding")
    if kind == "object":
        return _object_value(element)
    if kind == "array":
        if any(child.tag != "item" for child in element):
            raise ValueError("Array children must be item elements")
        return [_value(child) for child in element]
    text = _scalar_text(element)
    if kind == "null":
        if text.strip():
            raise ValueError("Null values cannot contain text")
        return None
    if kind == "boolean":
        if text not in ("true", "false"):
            raise ValueError("Boolean values must be true or false")
        return text == "true"
    if kind == "integer":
        if re.fullmatch(r"[+-]?[0-9]+", text) is None:
            raise ValueError("Integer values must contain decimal digits")
        try:
            return int(text)
        except ValueError as error:
            raise ValueError("Invalid integer value") from error
    if kind == "number":
        try:
            number = float(text)
        except ValueError as error:
            raise ValueError("Invalid number value") from error
        if not math.isfinite(number):
            raise ValueError("Number values must be finite")
        return number
    if kind != "string" and not (kind is None and element.tag == "key"):
        raise ValueError(f"Unsupported XML value type: {kind!r}")
    if encoding == "base64":
        try:
            return base64.b64decode(text, validate=True).decode(
                "utf-8", "surrogatepass"
            )
        except (ValueError, binascii.Error, UnicodeDecodeError) as error:
            raise ValueError("Invalid base64 UTF-8 string") from error
    return text


def _object_value(element: ET.Element) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for child in element:
        if child.get("type") is None:
            if child.tag != "field" or [node.tag for node in child] != ["key", "value"]:
                raise ValueError("Untyped object fields must contain a key and a value")
            key = _value(child.find("key"))
            if not isinstance(key, str):
                raise ValueError("Object keys must be strings")
            value = _value(child.find("value"))
        else:
            key, value = child.tag, _value(child)
        if key in result:
            raise ValueError(f"Duplicate object key: {key!r}")
        result[key] = value
    return result


def _scalar_text(element: ET.Element) -> str:
    if len(element):
        raise ValueError("Scalar values cannot contain child elements")
    if element.text is None:
        return ""
    if not isinstance(element.text, str):
        raise TypeError("XML element text must be a string or None")
    return element.text
