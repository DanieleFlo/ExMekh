from typing import Any, cast

import pytest

from qs.core.models.xml_header import XmlHeader


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


def test_prepend_to() -> None:
    header = XmlHeader(source="Tool", name="echo", id="id-1")
    result = header.prepend_to('{"result": "success"}')
    assert result == f'{header.to_xml()}\n{{"result": "success"}}'
