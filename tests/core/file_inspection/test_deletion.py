"""Batch deletion keeps per-path outcomes and confines recursive removal."""

import os

import pytest

from exmekh_lib.core.agents.capabilities.capabilities import Capabilities
from exmekh_lib.core.agents.capabilities.defaults import DefaultCapabilities
from exmekh_lib.core.file_inspection import edit_filesystem
from exmekh_lib.core.providers.openai_chat_completion_annotation_schema_builder import (
    OpenAIChatCompletionAnnotationSchemaBuilder,
)
from exmekh_lib.core.providers.openai_function_tool_definition_mapper import (
    OpenAIFunctionToolDefinitionMapper,
)


@pytest.fixture
def delete_tool(tmp_path):
    loaded = Capabilities(None, default_capabilities=DefaultCapabilities()).load(
        work_root=tmp_path
    )
    return next(tool for tool in loaded.tools if tool.name == "delete-file")


@pytest.mark.parametrize("strict", [False, True])
def test_delete_schema_requires_a_list_of_paths(delete_tool, strict):
    mapper = OpenAIFunctionToolDefinitionMapper(
        OpenAIChatCompletionAnnotationSchemaBuilder()
    )
    schema = mapper.map(delete_tool, strict=strict).get_parameters()

    assert schema["required"] == ["paths"]
    assert schema["properties"] == {
        "paths": {"type": "array", "items": {"type": "string"}}
    }


def test_delete_accepts_files_empty_directories_and_nested_directories(
    tmp_path, delete_tool
):
    (tmp_path / "file.bin").write_bytes(b"\x00\xff")
    (tmp_path / "empty").mkdir()
    nested = tmp_path / "folder" / "nested"
    nested.mkdir(parents=True)
    (nested / "content.txt").write_text("remove")

    paths = ["file.bin", "empty", "folder"]
    result = delete_tool.execute(paths=paths)

    assert result["ok"] is True
    assert [item["path"] for item in result["results"]] == paths
    assert all(
        item["ok"] and item["operation"] == "remove" for item in result["results"]
    )
    assert list(tmp_path.iterdir()) == []


def test_delete_continues_after_failure_and_reports_each_cause(tmp_path, delete_tool):
    (tmp_path / "first.txt").write_text("remove")
    (tmp_path / "last.txt").write_text("remove")
    paths = ["first.txt", "missing<&>.txt", "../outside", "last.txt"]

    result = delete_tool.execute(paths=paths)

    assert result["ok"] is False
    assert [item["path"] for item in result["results"]] == paths
    assert [item["ok"] for item in result["results"]] == [True, False, False, True]
    for item, code in zip(
        result["results"][1:3], ["file_not_found", "path_outside_workroot"]
    ):
        assert item["error_kind"] == "invalid_file_deletion"
        assert item["error_code"] == code
        assert item["error"] and item["hint"]
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("paths", [[], "notes.txt", None, {}])
def test_delete_requires_a_nonempty_list(tmp_path, delete_tool, paths):
    path = tmp_path / "notes.txt"
    path.write_text("KEEP")

    result = delete_tool.execute(paths=paths)

    assert result["ok"] is False
    assert result["error_code"] == "invalid_input"
    assert result["error"] and result["hint"]
    assert path.read_text() == "KEEP"


@pytest.mark.parametrize("invalid", ["", " ", "bad\x00path", None, 42])
def test_delete_invalid_entries_do_not_prevent_other_deletions(
    tmp_path, delete_tool, invalid
):
    path = tmp_path / "notes.txt"
    path.write_text("remove")

    result = delete_tool.execute(paths=[invalid, path.name])

    assert result["ok"] is False
    assert result["results"][0]["path"] == invalid
    assert result["results"][0]["error_code"] == "invalid_path"
    assert result["results"][1]["ok"] is True
    assert not path.exists()


@pytest.mark.parametrize("resource", [".", "folder/..", "root-alias"])
def test_delete_never_removes_the_workroot(tmp_path, delete_tool, resource):
    (tmp_path / "folder").mkdir()
    (tmp_path / "root-alias").symlink_to(tmp_path, target_is_directory=True)
    keep = tmp_path / "keep.txt"
    keep.write_text("KEEP")

    result = delete_tool.execute(paths=[resource])

    assert result["ok"] is False
    assert result["results"][0]["error_code"] == "invalid_path"
    assert keep.read_text() == "KEEP"


def test_delete_requires_relative_paths_even_inside_workroot(tmp_path, delete_tool):
    path = tmp_path / "notes.txt"
    path.write_text("KEEP")

    result = delete_tool.execute(paths=[str(path)])

    assert result["results"][0]["error_code"] == "invalid_path"
    assert path.read_text() == "KEEP"


def test_delete_rejects_outside_symlink_but_does_not_follow_nested_symlinks(
    tmp_path, tmp_path_factory, delete_tool
):
    outside = tmp_path_factory.mktemp("outside-deletion")
    keep = outside / "keep.txt"
    keep.write_text("KEEP")
    (tmp_path / "outside").symlink_to(outside, target_is_directory=True)
    folder = tmp_path / "folder"
    folder.mkdir()
    (folder / "outside").symlink_to(outside, target_is_directory=True)
    (folder / "dangling").symlink_to(outside / "missing")

    result = delete_tool.execute(paths=["outside", "folder"])

    assert result["results"][0]["error_code"] == "path_outside_workroot"
    assert result["results"][1]["ok"] is True
    assert not folder.exists()
    assert keep.read_text() == "KEEP"


def test_delete_permission_failure_does_not_stop_later_paths(
    tmp_path, monkeypatch, delete_tool
):
    denied = tmp_path / "denied"
    denied.mkdir()
    (denied / "keep.txt").write_text("KEEP")
    (tmp_path / "notes.txt").write_text("remove")

    def fail_rmtree(*args, **kwargs):
        raise PermissionError("directory deletion denied")

    monkeypatch.setattr(edit_filesystem.shutil, "rmtree", fail_rmtree)
    fail_rmtree.avoids_symlink_attacks = True

    result = delete_tool.execute(paths=["denied", "notes.txt"])

    assert result["results"][0]["error_code"] == "permission_denied"
    assert result["results"][0]["error"] == "directory deletion denied"
    assert result["results"][1]["ok"] is True
    assert (denied / "keep.txt").read_text() == "KEEP"
    assert not (tmp_path / "notes.txt").exists()


def test_delete_rejects_special_files_and_continues(tmp_path, delete_tool):
    os.mkfifo(tmp_path / "pipe")
    (tmp_path / "notes.txt").write_text("remove")

    result = delete_tool.execute(paths=["pipe", "notes.txt"])

    assert result["results"][0]["error_code"] == "invalid_deletion_target"
    assert result["results"][1]["ok"] is True
    assert (tmp_path / "pipe").exists()


def test_delete_refuses_unsafe_recursive_removal_but_still_deletes_files(
    tmp_path, monkeypatch, delete_tool
):
    folder = tmp_path / "folder"
    folder.mkdir()
    (tmp_path / "notes.txt").write_text("remove")
    monkeypatch.setattr(edit_filesystem.shutil.rmtree, "avoids_symlink_attacks", False)

    result = delete_tool.execute(paths=["folder", "notes.txt"])

    assert result["results"][0]["error_code"] == "unsupported_platform"
    assert result["results"][1]["ok"] is True
    assert folder.is_dir()


def test_delete_rechecks_directory_before_recursive_removal(
    tmp_path, tmp_path_factory, monkeypatch, delete_tool
):
    folder = tmp_path / "folder"
    folder.mkdir()
    (folder / "keep.txt").write_text("ORIGINAL")
    outside = tmp_path_factory.mktemp("redirected-deletion")
    (outside / "keep.txt").write_text("OUTSIDE")
    open_parent = edit_filesystem._open_parent_directory

    def redirect_directory(root, target):
        descriptor = open_parent(root, target)
        target.rename(tmp_path / "original")
        target.symlink_to(outside, target_is_directory=True)
        return descriptor

    monkeypatch.setattr(edit_filesystem, "_open_parent_directory", redirect_directory)

    result = delete_tool.execute(paths=["folder"])

    assert result["ok"] is False
    assert (tmp_path / "original" / "keep.txt").read_text() == "ORIGINAL"
    assert (outside / "keep.txt").read_text() == "OUTSIDE"


@pytest.mark.parametrize("directory", [False, True])
def test_delete_does_not_remove_the_target_of_an_internal_symlink(
    tmp_path, delete_tool, directory
):
    target = tmp_path / "target"
    if directory:
        target.mkdir()
        (target / "keep.txt").write_text("KEEP")
    else:
        target.write_text("KEEP")
    alias = tmp_path / "alias"
    alias.symlink_to(target, target_is_directory=directory)
    result = delete_tool.execute(paths=["alias"])
    assert result["results"][0]["error_code"] == "invalid_deletion_target"
    assert alias.is_symlink()
    assert target.exists()
