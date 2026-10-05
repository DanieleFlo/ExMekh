"""Regressions for consistent navigation, limits and dependency errors."""

import builtins
import io
import tarfile
from dataclasses import replace
from pathlib import Path

import pytest

from exmekh_lib.config.file_inspection import FILE_INSPECTION
from exmekh_lib.core.agents.capabilities.capabilities import Capabilities
from exmekh_lib.core.agents.capabilities.defaults import DefaultCapabilities
from exmekh_lib.core.file_inspection import selection
from exmekh_lib.core.file_inspection.service import inspect_file


def test_markdown_index_limit_keeps_the_first_document_region(tmp_path: Path) -> None:
    path = tmp_path / "notes.md"
    path.write_text("# Intro\nbody\n```python\nx = 1\n```\n")
    result = inspect_file(
        tmp_path, path.name, settings=replace(FILE_INSPECTION, max_regions=1)
    )
    assert result["regions"] == [
        {"kind": "section", "name": "Intro", "start_line": 1, "end_line": 5}
    ]
    assert result["selection"]["index_truncated"]


def test_xml_index_limit_keeps_the_root_before_children(tmp_path: Path) -> None:
    path = tmp_path / "doc.xml"
    path.write_text("<root>\n<child/>\n</root>\n")
    result = inspect_file(
        tmp_path, path.name, settings=replace(FILE_INSPECTION, max_regions=1)
    )
    assert result["regions"] == [
        {"kind": "element", "name": "root", "start_line": 1, "end_line": 3}
    ]
    assert result["selection"]["index_truncated"]


@pytest.mark.parametrize("count", [1, 2])
def test_tar_limit_is_partial_only_when_a_member_is_omitted(
    tmp_path: Path, count: int
) -> None:
    path = tmp_path / "data.tar"
    with tarfile.open(path, "w") as archive:
        for number in range(count):
            info = tarfile.TarInfo(f"item-{number}")
            info.size = 1
            archive.addfile(info, io.BytesIO(b"x"))
    result = inspect_file(
        tmp_path, path.name, settings=replace(FILE_INSPECTION, max_archive_entries=1)
    )
    assert result["analysis_status"] == ("complete" if count == 1 else "skipped_limit")
    assert result["selection"]["index_truncated"] is (count > 1)
    assert result["analysis"]["entries_scanned"] == 1


@pytest.mark.parametrize(
    ("filename", "content"),
    [
        ("data.json", '{"a": "first\u2028second", "b": 2}'),
        ("data.yaml", 'a: "first\u0085second"\nb: 2\n'),
        ("data.yaml", 'a: "first\u2028second"\nb: 2\n'),
    ],
)
def test_parser_marks_use_physical_cr_lf_lines(
    tmp_path: Path, filename: str, content: str
) -> None:
    path = tmp_path / filename
    path.write_text(content)
    result = inspect_file(tmp_path, filename)
    assert all(
        1 <= region["start_line"] <= region["end_line"] <= result["total_lines"]
        for region in result["regions"]
    )
    expected_line = 1 if filename.endswith(".json") else 2
    assert (
        next(region for region in result["regions"] if region["name"] == "/b")[
            "start_line"
        ]
        == expected_line
    )


def test_excessive_query_is_rejected_before_tokenization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_if_tokenized(value: str) -> set[str]:
        raise AssertionError("Oversized query should not be tokenized")

    monkeypatch.setattr(selection, "query_terms", fail_if_tokenized)
    with pytest.raises(ValueError, match="at most"):
        inspect_file(
            tmp_path, "missing", query="a" * (FILE_INSPECTION.max_query_chars + 1)
        )


def test_file_tool_reports_missing_dependencies_without_breaking_catalog_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import_original = builtins.__import__

    def missing_service(name, *args, **kwargs):
        if name == "exmekh_lib.core.file_inspection.service":
            raise ModuleNotFoundError("unavailable", name="pypdf")
        return import_original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_service)
    loaded = Capabilities(None, default_capabilities=DefaultCapabilities()).load(
        work_root=tmp_path
    )
    tool = next(tool for tool in loaded.tools if tool.name == "inspect-file")
    result = tool.execute(path="report.pdf")
    assert result["ok"] is False
    assert result["error_kind"] == "missing_dependency"
    assert "pypdf" in result["error"]


@pytest.mark.parametrize("prefix", ["# Title\n", "```\ncode\n```\n"])
def test_setext_heading_does_not_reuse_atx_or_fence_lines(
    tmp_path: Path, prefix: str
) -> None:
    path = tmp_path / "notes.md"
    path.write_text(prefix + "===\n")
    result = inspect_file(tmp_path, path.name)
    assert all(
        region["start_line"] <= region["end_line"] for region in result["regions"]
    )
    assert not any(region["name"] in {"# Title", "```"} for region in result["regions"])


def test_corrupt_xz_tar_is_an_invalid_format_result(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.tar.xz"
    path.write_bytes(b"\xfd7zXZ\x00bad archive")
    result = inspect_file(tmp_path, path.name)
    assert result["analysis_status"] == "invalid_format"
