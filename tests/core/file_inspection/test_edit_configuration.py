"""Changed marker settings must reach selectors, prompts, schemas and SDK calls."""

from dataclasses import replace

import pytest

from exmekh_lib.config import file_editing
from exmekh_lib.core.file_inspection import edit_filesystem, edit_selection, editing
from exmekh_lib.core.file_inspection.errors import FileInspectionError
from tests.interactive_support import (
    enter,
    exit_skill,
    inspect,
    make_sdk_node,
    pause,
    read,
    response,
)


@pytest.mark.parametrize(
    "prefix", ["@alternate.edit+", file_editing.FILE_EDITING.marker_prefix]
)
@pytest.mark.parametrize(
    "operation", ["create", "create_line", "edit", "insert", "remove"]
)
def test_changed_markers_are_used_throughout_the_sdk_node(
    tmp_path, monkeypatch, prefix, operation
):
    settings = replace(
        file_editing.FILE_EDITING,
        marker_prefix=prefix,
        all_marker_name="whole",
        delete_marker_name="erase",
        line_marker_name="before",
    )
    for module in (file_editing, edit_selection, editing, edit_filesystem):
        monkeypatch.setattr(module, "FILE_EDITING", settings)
    path = tmp_path / "notes.txt"
    existing = operation not in {"create", "create_line"}
    if existing:
        path.write_bytes(b"one\ntwo")
    old = {
        "create": settings.all_marker,
        "create_line": settings.first_line_marker,
        "edit": "one",
        "insert": settings.line_marker(2),
        "remove": settings.all_marker,
    }[operation]
    new = settings.delete_marker if operation == "remove" else "new\n"

    def resume(body):
        configuration = server.requests[-2]["agentic"]["interactive"]["tools"][0]
        assert configuration["markers"] == {
            "line": settings.line_family,
            "all": settings.all_marker,
            "delete": settings.delete_marker,
        }
        prompt = str(server.requests[-2]["input"])
        for marker in (
            settings.all_marker,
            settings.delete_marker,
            settings.line_example,
        ):
            assert marker in prompt
        wire_tool = next(
            tool for tool in server.requests[-2]["tools"] if tool["name"] == "edit-file"
        )
        assert (
            settings.all_marker
            in wire_tool["parameters"]["properties"]["old_text"]["description"]
        )
        assert (
            settings.delete_marker
            in wire_tool["parameters"]["properties"]["new_text"]["description"]
        )
        assert body["agentic"]["interactive"]["reply"]["status"] == (
            "provide" if existing else "create"
        )
        return response(
            (
                "edit",
                "edit-file",
                {"path": path.name, "old_text": old, "new_text": new},
            ),
            continuation="cont-edit",
        )

    steps = [enter()] + ([inspect(), read(1, 2)] if existing else [])
    steps += [pause(), resume, exit_skill(), response(text="done")]
    node, server, provider = make_sdk_node(tmp_path, monkeypatch, steps)
    try:
        node.run("change file")
        if operation == "remove":
            assert not path.exists()
        else:
            assert (
                path.read_bytes()
                == {
                    "create": b"new\n",
                    "create_line": b"new\n",
                    "edit": b"new\n\ntwo",
                    "insert": b"one\nnew\ntwo",
                }[operation]
            )
    finally:
        provider.get_session().close()
        server.client.close()


def test_changed_marker_errors_and_reserved_family_follow_the_configuration(
    monkeypatch,
):
    settings = replace(file_editing.FILE_EDITING, marker_prefix="%changed.[]+")
    monkeypatch.setattr(edit_selection, "FILE_EDITING", settings)
    for malformed in (
        settings.marker_prefix,
        settings.line_marker("01"),
        " " + settings.all_marker,
    ):
        with pytest.raises(FileInspectionError) as error:
            edit_selection.select_replacement("one", malformed, "new", exists=True)
        assert settings.reserved_prefix in str(error.value)
    with pytest.raises(FileInspectionError) as error:
        edit_selection.select_replacement("", "literal", "new", exists=False)
    assert settings.all_marker in str(error.value)
    assert settings.first_line_marker in str(error.value)
