import sys
from xml.etree import ElementTree as ET

import pytest

from tests.tool_result_support import _value, parse_tool_result


@pytest.mark.parametrize(
    "body",
    [
        '<tool-result type="object"><field><key>x</key></field></tool-result>',
        '<tool-result type="object"><field><value type="null" /></field></tool-result>',
        '<tool-result type="object"><field><key type="integer">1</key>'
        '<value type="null" /></field></tool-result>',
        '<tool-result type="object"><x type="null" /><x type="null" /></tool-result>',
        '<tool-result type="object"><untyped /></tool-result>',
        '<tool-result type="array"><value type="null" /></tool-result>',
        '<tool-result type="boolean">yes</tool-result>',
        '<tool-result type="boolean" />',
        '<tool-result type="integer" />',
        '<tool-result type="integer">1.2</tool-result>',
        '<tool-result type="integer">1_000</tool-result>',
        '<tool-result type="number" />',
        '<tool-result type="number">not a number</tool-result>',
        '<tool-result type="number">nan</tool-result>',
        '<tool-result type="number">inf</tool-result>',
        '<tool-result type="null">text</tool-result>',
        '<tool-result type="string"><child /></tool-result>',
        '<tool-result type="unknown">text</tool-result>',
        "<tool-result>text</tool-result>",
        '<tool-result type="string" encoding="hex">00</tool-result>',
        '<tool-result type="integer" encoding="base64">MQ==</tool-result>',
        '<tool-result type="string" encoding="base64">!!!!</tool-result>',
        '<tool-result type="string" encoding="base64">/w==</tool-result>',
    ],
)
def test_malformed_typed_results_raise_value_error(body: str) -> None:
    with pytest.raises(ValueError):
        parse_tool_result(body)


def test_missing_element_raises_explicit_error() -> None:
    with pytest.raises(ValueError, match="Missing XML value element"):
        _value(None)


def test_non_element_input_raises_type_error() -> None:
    with pytest.raises(TypeError, match="must be an Element"):
        _value("text")  # type: ignore[arg-type]


def test_non_string_element_text_raises_type_error() -> None:
    element = ET.Element("value", {"type": "string"})
    element.text = 123  # type: ignore[assignment]
    with pytest.raises(TypeError, match="text must be a string"):
        _value(element)


def test_non_string_content_raises_type_error() -> None:
    with pytest.raises(TypeError, match="content must be a string"):
        parse_tool_result(None)  # type: ignore[arg-type]


def test_integer_conversion_error_preserves_original_cause() -> None:
    digit_limit = sys.get_int_max_str_digits()
    if digit_limit == 0:
        pytest.skip("Integer conversion digit limit is disabled")
    text = "9" * (digit_limit + 1)
    with pytest.raises(ValueError, match="Invalid integer value") as caught:
        parse_tool_result(f'<tool-result type="integer">{text}</tool-result>')
    assert isinstance(caught.value.__cause__, ValueError)


def test_number_conversion_error_preserves_original_cause() -> None:
    with pytest.raises(ValueError, match="Invalid number value") as caught:
        parse_tool_result('<tool-result type="number">invalid</tool-result>')
    assert isinstance(caught.value.__cause__, ValueError)
