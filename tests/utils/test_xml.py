import base64
from xml.etree import ElementTree as ET

import pytest

from qs.utils.xml import render_xml, stringify_tool_result
from tests.tool_result_support import parse_tool_result


def test_structured_result_preserves_nested_data_and_escapes_page_content():
    markdown = '# Wikipedia\nÈ un testo con <header> & "citazioni".'
    root = ET.fromstring(
        stringify_tool_result(
            {
                "markdown": markdown,
                "page_dates": {"published": None},
                "untrusted_content": True,
                "status_code": 200,
                "elapsed_seconds": 8.37,
                "results": [{"title": "Wikipedia"}, "next"],
                "empty": [],
            }
        )
    )
    assert root.tag == "tool-result"
    assert root.findtext("markdown") == markdown
    assert root.find("page_dates/published").get("type") == "null"
    assert root.findtext("untrusted_content") == "true"
    assert root.find("status_code").get("type") == "integer"
    assert root.findtext("elapsed_seconds") == "8.37"
    assert root.findtext("results/item/title") == "Wikipedia"
    assert root.findall("results/item")[1].text == "next"
    assert root.find("empty").get("type") == "array"


def test_invalid_keys_and_control_characters_remain_recoverable():
    key = "bad key <&\x00"
    value = "content\x00\ud800"
    root = ET.fromstring(stringify_tool_result({key: value}))
    field = root.find("field")
    for element, original in ((field.find("key"), key), (field.find("value"), value)):
        assert element.get("encoding") == "base64"
        assert (
            base64.b64decode(element.text).decode("utf-8", "surrogatepass") == original
        )


@pytest.mark.parametrize("text", ["plain text", "<custom>XML</custom>", '{"a": 1}'])
def test_tool_authored_text_is_preserved(text):
    assert stringify_tool_result(text) == text


def test_cycles_and_non_json_values_do_not_break_xml_rendering():
    value = {"tuple": (1, False), "infinite": float("inf"), "unsupported": 1j}
    value["cycle"] = value
    root = ET.fromstring(stringify_tool_result(value))
    assert root.findtext("tuple/item") == "1"
    assert root.findtext("infinite") == "inf"
    assert root.findtext("unsupported") == "1j"
    assert root.findtext("cycle") == "<circular reference>"


def test_typed_result_can_be_decoded_without_losing_empty_values_or_line_endings():
    value = {
        "empty": "",
        "null": None,
        "boolean": False,
        "array": [],
        "object": {},
        "field": "ordinary field name",
        "bad key\x00": "invalid\x00",
        "markdown": "# Title\r\nparagraph\rtail",
        "integer": 200,
        "number": 8.37,
        "nested": [{"key": "value"}],
    }
    assert parse_tool_result(stringify_tool_result(value)) == value


def test_rendering_does_not_mutate_the_callers_xml_tree():
    root = ET.Element("result")
    ET.SubElement(root, "text").text = "first\r\nsecond"
    original = ET.tostring(root, encoding="unicode")

    rendered = render_xml(root)

    assert ET.fromstring(rendered).findtext("text") == "first\r\nsecond"
    assert ET.tostring(root, encoding="unicode") == original
