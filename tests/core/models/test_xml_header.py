import base64
from typing import Any, cast
from xml.etree import ElementTree as ET

import pytest

from qs.core.models.xml_header import XmlHeader
from qs.utils.xml import render_xml
from tests.tool_result_support import parse_tool_result


def test_init_valid() -> None:
    tool_header = XmlHeader(source="Tool", name="  search  ", id="  tool-1  ")
    assert tool_header.source == "Tool"
    assert tool_header.name == "search"
    assert tool_header.id == "tool-1"
    assert tool_header.fields == {}

    skill_header = XmlHeader(
        source="DeepSkill",
        name="coding",
        id="skill-1",
        fields={"depth": 1},
    )
    assert skill_header.source == "DeepSkill"
    assert skill_header.name == "coding"
    assert skill_header.id == "skill-1"
    assert skill_header.fields == {"depth": 1}


def test_init_rejects_non_string_source() -> None:
    with pytest.raises(TypeError, match="source must be a string"):
        XmlHeader(source=cast(Any, 123), name="echo", id="id-1")


def test_init_rejects_invalid_source_value() -> None:
    with pytest.raises(ValueError, match="source must be 'Tool' or 'DeepSkill'"):
        XmlHeader(source=cast(Any, "Other"), name="echo", id="id-1")


def test_init_rejects_non_string_name() -> None:
    with pytest.raises(TypeError, match="name must be a string"):
        XmlHeader(source="Tool", name=cast(Any, 123), id="id-1")


@pytest.mark.parametrize("empty_name", ["", "   ", "\t\n"])
def test_init_rejects_empty_name(empty_name: str) -> None:
    with pytest.raises(ValueError, match="name cannot be empty"):
        XmlHeader(source="Tool", name=empty_name, id="id-1")


def test_init_rejects_non_string_id() -> None:
    with pytest.raises(TypeError, match="id must be a string"):
        XmlHeader(source="Tool", name="echo", id=cast(Any, 123))


@pytest.mark.parametrize("empty_id", ["", "   ", "\t\n"])
def test_init_rejects_empty_id(empty_id: str) -> None:
    with pytest.raises(ValueError, match="id cannot be empty"):
        XmlHeader(source="Tool", name="echo", id=empty_id)


def test_init_rejects_non_dict_fields() -> None:
    with pytest.raises(TypeError, match="fields must be a dictionary"):
        XmlHeader(source="Tool", name="echo", id="id-1", fields=cast(Any, [1, 2]))


def test_init_rejects_non_string_field_keys() -> None:
    with pytest.raises(TypeError, match="all keys in fields must be strings"):
        XmlHeader(source="Tool", name="echo", id="id-1", fields=cast(Any, {123: "val"}))


def test_add_field_and_chaining() -> None:
    header = XmlHeader(source="Tool", name="echo", id="id-1")
    ret = header.add_field("retries", 3)
    assert ret is header
    assert header.fields["retries"] == 3

    header.add_field("retries", 5)
    assert header.fields["retries"] == 5


def test_add_fields_and_chaining() -> None:
    header = XmlHeader(source="Tool", name="echo", id="id-1")
    ret = header.add_fields(depth=2, tag="analysis")
    assert ret is header
    assert header.fields == {"depth": 2, "tag": "analysis"}


def test_remove_field() -> None:
    header = XmlHeader(source="Tool", name="echo", id="id-1", fields={"key": "val"})
    ret = header.remove_field("key")
    assert ret is header
    assert "key" not in header.fields

    # Removing non-existent key does not raise
    ret = header.remove_field("non_existent")
    assert ret is header


def test_to_xml_without_fields() -> None:
    header = XmlHeader(source="Tool", name="echo", id="id-1")
    expected = (
        "<header>\n  <name>echo</name>\n  <type>Tool</type>\n  <id>id-1</id>\n</header>"
    )
    assert header.to_xml() == expected


def test_to_xml_with_fields_and_none_filtering() -> None:
    header = XmlHeader(
        source="DeepSkill",
        name="coding",
        id="skill-1",
        fields={"depth": 2, "ignored": None, "active": True},
    )
    xml_output = header.to_xml()
    expected = (
        "<header>\n"
        "  <name>coding</name>\n"
        "  <type>DeepSkill</type>\n"
        "  <id>skill-1</id>\n"
        "  <depth>2</depth>\n"
        "  <active>True</active>\n"
        "</header>"
    )
    assert xml_output == expected
    assert "ignored" not in xml_output


@pytest.mark.parametrize("as_list", [True, False])
def test_structured_metadata_preserves_types_and_escapes_nested_text(as_list):
    record = {
        "name": "read <page> & text",
        "tool-call-id": "call\x00\ud800",
        "output-id": "output\r\nid",
        "depth": 3,
        "active": True,
        "missing": None,
        "bad key": [],
    }
    value = [record] if as_list else record
    header = XmlHeader(source="DeepSkill", name="skill", id="result").add_field(
        "background-tools", value
    )
    root = ET.fromstring(header.to_xml())
    metadata = root.find("background-tools")

    assert metadata is not None
    assert parse_tool_result(render_xml(metadata)) == value


def test_prepend_to() -> None:
    header = XmlHeader(source="Tool", name="echo", id="id-1")
    result = header.prepend_to('{"result": "success"}')
    assert result == f'{header.to_xml()}\n{{"result": "success"}}'


def test_to_xml_renders_typed_arguments_and_escapes_content() -> None:
    header = XmlHeader(
        source="Tool",
        name="inspect",
        id="result-1",
        args={
            "text": 'city <Rome> & "Lazio"',
            "count": 2,
            "active": True,
            "missing": None,
            "options": {"lines": [1, 20], "encoding": "utf-8"},
        },
    )

    assert header.to_xml() == (
        "<header>\n"
        "  <name>inspect</name>\n"
        "  <type>Tool</type>\n"
        "  <id>result-1</id>\n"
        "  <args>\n"
        '    <text>city &lt;Rome&gt; &amp; "Lazio"</text>\n'
        "    <count>2</count>\n"
        "    <active>true</active>\n"
        "    <missing>null</missing>\n"
        '    <options>{"lines":[1,20],"encoding":"utf-8"}</options>\n'
        "  </args>\n"
        "</header>"
    )


def test_rejects_non_dictionary_arguments() -> None:
    with pytest.raises(TypeError, match="args must be a dictionary"):
        XmlHeader(
            source="Tool",
            name="inspect",
            id="result-1",
            args=cast(Any, []),
        )


def test_invalid_xml_argument_name_uses_escaped_generic_node() -> None:
    header = XmlHeader(
        source="DeepSkill",
        name="skill",
        id="result-1",
        args={'bad key & "quoted"': "value"},
    )

    assert '<arg name="bad key &amp; &quot;quoted&quot;">value</arg>' in header.to_xml()


def test_rejects_non_string_argument_names() -> None:
    with pytest.raises(TypeError, match="all keys in args must be strings"):
        XmlHeader(
            source="Tool",
            name="inspect",
            id="result-1",
            args=cast(dict[str, Any], {1: "value"}),
        )


def test_leading_header_requires_a_complete_well_formed_xml_block() -> None:
    assert not XmlHeader.has_leading_header("<header><name>missing close")
    assert not XmlHeader.has_leading_header("<header><name></header>")


def test_header_text_preserves_control_characters_and_line_endings() -> None:
    text = "reasoning\x00\ud800"
    argument = "first\r\nsecond\rtail"
    header = XmlHeader(
        source="Tool", name="inspect", id="output", args={"text": argument}
    ).add_field("thinking", text)

    root = ET.fromstring(header.to_xml())

    assert root.findtext("args/text") == argument
    thinking = root.find("thinking")
    assert thinking.get("encoding") == "base64"
    assert base64.b64decode(thinking.text).decode("utf-8", "surrogatepass") == text


def test_annotation_preserves_the_header_and_tool_authored_body() -> None:
    body = "# Tool output\r\n<custom>unchanged</custom>"
    header = XmlHeader(source="Tool", name="inspect", id="output")
    content = header.prepend_to(body)
    annotation = "reasoning\x00\ud800"

    annotated = XmlHeader.add_field_to_content(content, "thinking", annotation)

    header_text, remaining = annotated.split("</header>\n", 1)
    root = ET.fromstring(header_text + "</header>")
    assert remaining == body
    thinking = root.find("thinking")
    assert thinking.get("encoding") == "base64"
    assert (
        base64.b64decode(thinking.text).decode("utf-8", "surrogatepass") == annotation
    )


def test_argument_names_with_control_characters_use_encoded_attributes() -> None:
    argument_name = "name\x00\ud800"
    header = XmlHeader(
        source="Tool", name="inspect", id="output", args={argument_name: "value"}
    )

    root = ET.fromstring(header.to_xml())

    argument = root.find("args/arg")
    assert argument.text == "value"
    assert argument.get("name-encoding") == "base64"
    assert (
        base64.b64decode(argument.get("name")).decode("utf-8", "surrogatepass")
        == argument_name
    )
