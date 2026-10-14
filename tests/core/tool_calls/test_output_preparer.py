from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock
from xml.etree import ElementTree as ET

import pytest

from exmekh_lib.config.file_inspection import FILE_INSPECTION
from exmekh_lib.config.tool_output import ToolOutputSettings
from exmekh_lib.core.cache import SessionCache
from exmekh_lib.core.cache.output_file_workspace import OutputFileWorkspace
from exmekh_lib.core.cache.tool_call_record import ToolCallRecord
from exmekh_lib.core.file_inspection.errors import FileInspectionError
from exmekh_lib.core.file_inspection.output_access import output_file_access
from exmekh_lib.core.file_inspection.workspace import read_snapshot
from exmekh_lib.core.models.tool import PreformattedToolResult
from exmekh_lib.core.models.tool_output_presentation import ToolOutputPresentation
from exmekh_lib.core.models.tool_output_reference import ToolOutputReference
from exmekh_lib.core.models.xml_header import XmlHeader
from exmekh_lib.core.tool_calls.output_preparer import ToolOutputPreparer
from exmekh_lib.core.tool_calls.output_view import ToolOutputView
from exmekh_lib.interfaces.tokenizer import Tokenizer


@pytest.fixture
def preparer(tmp_path):
    tokenizer = MagicMock(spec=Tokenizer)
    tokenizer.count_tokens.side_effect = len
    tokenizer.truncate_text.side_effect = lambda text, count: text[:count]
    tokenizer.split_text.side_effect = lambda text, count: Tokenizer.split_text(
        tokenizer, text, count
    )
    with SessionCache(tmp_path / "cache") as session:
        cache = session.node("node")
        tokenizer.get_session.return_value = session
        yield (
            ToolOutputPreparer(
                tokenizer,
                OutputFileWorkspace(
                    lambda: cache, tmp_path, ToolOutputSettings(max_tokens=10)
                ),
            ),
            cache,
            tokenizer,
        )


@pytest.mark.parametrize("count", [9, 10, 11])
def test_budget_excludes_header_and_only_exports_above_threshold(preparer, count):
    service, cache, _ = preparer
    text = "a" * count
    content = (
        XmlHeader("Tool", "work", "id")
        .add_field("args", "large" * 100)
        .prepend_to(text)
    )
    result = service.prepare(
        ToolOutputPresentation(
            ToolOutputReference("work", "id"), PreformattedToolResult(content)
        )
    )
    if count <= 10:
        assert result.content == content
        assert result.external_output is None
        assert not service.files.root.exists()
    else:
        reference = result.external_output
        assert reference is not None
        assert reference.token_count == count
        assert [item.token_count for item in reference.files] == [10, 1]
        assert text not in result.content
        assert "large" * 100 in result.content
        assert cache.get_all() == []


@pytest.mark.parametrize(
    "text", ["abcdefghijklmno", "é😀漢字" * 8, "12345678901234567890\r\n"]
)
def test_single_physical_line_is_cut_losslessly_in_order(preparer, tmp_path, text):
    service, _, _ = preparer
    result = service.prepare(
        ToolOutputPresentation(
            ToolOutputReference("work", "id"), PreformattedToolResult(text)
        )
    )
    reference = result.external_output
    assert reference is not None
    parts = [
        (tmp_path / item.path).read_bytes().decode("utf-8") for item in reference.files
    ]
    assert "".join(parts) == text
    assert all(item.token_count <= 10 for item in reference.files)
    assert reference.line_count == 1
    assert [item.path for item in reference.files] == [
        node.text
        for node in ET.fromstring(XmlHeader.body_from_content(result.content)).findall(
            "files/item/path"
        )
    ]


@pytest.mark.parametrize("separator", ["\n", "\r\n", "\r"])
def test_multiline_output_stays_in_one_file_and_preserves_newlines(
    preparer, tmp_path, separator
):
    service, _, _ = preparer
    text = (
        separator.join(["123456789012345", "Markdown <&> with spaces  ", "last"])
        + separator
    )
    result = service.prepare(
        ToolOutputPresentation(
            ToolOutputReference("work", "id"), PreformattedToolResult(text)
        )
    )
    reference = result.external_output
    assert reference is not None
    assert len(reference.files) == 1
    assert reference.line_count == 3
    assert (tmp_path / reference.files[0].path).read_bytes() == text.encode("utf-8")


def test_repeated_and_concurrent_reads_reuse_files_and_reused_ids_do_not_collide(
    preparer,
):
    service, _, _ = preparer
    with ThreadPoolExecutor(max_workers=4) as workers:
        results = list(
            workers.map(
                lambda _: service.prepare(
                    ToolOutputPresentation(
                        ToolOutputReference("work", "same"),
                        PreformattedToolResult("x" * 25),
                    )
                ),
                range(20),
            )
        )
    assert all(
        result.external_output == results[0].external_output for result in results
    )
    another = service.prepare(
        ToolOutputPresentation(
            ToolOutputReference("work", "same"), PreformattedToolResult("y" * 25)
        )
    )
    assert another.external_output.files != results[0].external_output.files


def test_cache_roundtrip_and_every_view_return_reference_without_original(preparer):
    service, cache, _ = preparer
    text = "SECRET" * 20
    result = service.prepare(
        ToolOutputPresentation(
            ToolOutputReference("work", "id"), PreformattedToolResult(text)
        )
    )
    record = ToolCallRecord("work", "id", {}, text).externalized(result.external_output)
    cache.store(record)
    restored = cache.get("id")
    assert restored.result is None
    assert restored.external_output == result.external_output
    view = ToolOutputView(restored)
    for content in [
        view.content(),
        view.format_saved(None),
        view.format_saved(999),
        view.format_embedded(),
        view.preview(240),
    ]:
        assert text not in content
    assert view.format_saved(999) == view.format_saved(None)


@pytest.mark.parametrize("failure", ["count", "prefix", "write"])
def test_failures_return_explicit_xml_without_original_or_partial_files(
    preparer, monkeypatch, failure
):
    service, cache, tokenizer = preparer
    if failure == "count":
        tokenizer.count_tokens.side_effect = OSError("count unavailable")
    elif failure == "prefix":
        tokenizer.truncate_text.side_effect = lambda text, count: "not a prefix"
    else:
        monkeypatch.setattr(
            Path, "write_bytes", MagicMock(side_effect=OSError("disk full"))
        )
    result = service.prepare(
        ToolOutputPresentation(
            ToolOutputReference("work", "id"), PreformattedToolResult("secret" * 20)
        )
    )
    assert result.failure["error_kind"] == "tool_output_export_failed"
    assert "secret" * 20 not in result.content
    assert result.external_output is None
    assert not list(service.files.root.rglob("*.txt"))
    assert cache.get_all() == []


def test_reference_manifest_is_never_exported_again(preparer):
    service, _, tokenizer = preparer
    first = service.prepare(
        ToolOutputPresentation(
            ToolOutputReference("work", "id"), PreformattedToolResult("x" * 50)
        )
    )
    tokenizer.count_tokens.side_effect = AssertionError(
        "references must not be counted"
    )
    repeated = service.prepare(
        ToolOutputPresentation(
            ToolOutputReference("work", "id"),
            PreformattedToolResult(
                first.content, external_output=first.external_output
            ),
        )
    )
    assert repeated.content == first.content
    assert repeated.external_output == first.external_output


def test_file_reads_are_node_scoped_and_allow_large_multiline_artifacts(
    preparer, tmp_path
):
    service, cache, tokenizer = preparer
    first = service.prepare(
        ToolOutputPresentation(
            ToolOutputReference("work", "id"), PreformattedToolResult("large\n" * 20)
        )
    )
    path = first.external_output.files[0].path
    with output_file_access(service.files):
        snapshot = read_snapshot(
            tmp_path, path, replace(FILE_INSPECTION, max_file_bytes=2)
        )
        assert snapshot.content == b"large\n" * 20
    foreign = ToolOutputPreparer(
        tokenizer,
        OutputFileWorkspace(
            lambda: tokenizer.get_session().node("other"), tmp_path, service.settings
        ),
    )
    with (
        output_file_access(foreign.files),
        pytest.raises(FileInspectionError, match="not available"),
    ):
        read_snapshot(tmp_path, path, FILE_INSPECTION)


@pytest.mark.parametrize("limit", [0, -1, True, 1.2])
def test_config_rejects_invalid_budgets(limit):
    with pytest.raises(ValueError, match="positive integer"):
        ToolOutputSettings(max_tokens=limit)


@pytest.mark.parametrize("path", [Path(".."), Path("."), Path("/outside")])
def test_config_rejects_unsafe_directories(path):
    with pytest.raises(ValueError):
        ToolOutputSettings(directory=path)


def test_config_uses_environment(monkeypatch):
    monkeypatch.setenv("EXMEKH_LIB_TOOL_OUTPUT_MAX_TOKENS", "123")
    monkeypatch.setenv("EXMEKH_LIB_TOOL_OUTPUT_DIR", "artifacts/tool_text")
    assert ToolOutputSettings.from_environment() == ToolOutputSettings(
        123, Path("artifacts/tool_text")
    )
