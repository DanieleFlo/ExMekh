from dataclasses import replace
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest

from exmekh_lib.config import file_inspection
from exmekh_lib.core.agents.capability.loader import MarkdownLoader


def test_loads_markdown_file(tmp_path: Path) -> None:
    markdown_path = tmp_path / "skill.MD"
    markdown_path.write_text("content")

    assert MarkdownLoader(markdown_path).load() == "content"


def test_rejects_non_path() -> None:
    with pytest.raises(TypeError, match="markdown_path must be a Path"):
        MarkdownLoader(cast(Path, "skill.md"))


def test_rejects_missing_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        MarkdownLoader(tmp_path / "missing.md").load()


def test_rejects_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be a file"):
        MarkdownLoader(tmp_path).load()


def test_rejects_non_markdown_file(tmp_path: Path) -> None:
    text_path = tmp_path / "skill.txt"
    text_path.write_text("content")

    with pytest.raises(ValueError, match="must be a Markdown file"):
        MarkdownLoader(text_path).load()


def test_rejects_symbolic_link(tmp_path: Path) -> None:
    target = tmp_path / "target.md"
    target.write_text("content")
    link = tmp_path / "link.md"
    link.symlink_to(target)

    with pytest.raises(ValueError, match="must not be a symbolic link"):
        MarkdownLoader(link).load()


def test_propagates_file_read_error(tmp_path: Path) -> None:
    markdown_path = tmp_path / "skill.md"
    markdown_path.write_text("content")

    with (
        patch(
            "exmekh_lib.core.agents.capability.loader.read_file_from_path",
            side_effect=OSError("unreadable"),
        ),
        pytest.raises(OSError, match="unreadable"),
    ):
        MarkdownLoader(markdown_path).load()


def test_configuration_references_follow_changed_limits_and_preserve_other_braces(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        file_inspection,
        "FILE_INSPECTION",
        replace(
            file_inspection.FILE_INSPECTION, max_pdf_render_pages=2, pdf_render_dpi=72
        ),
    )
    markdown_path = tmp_path / "skill.md"
    markdown_path.write_text(
        "Pages: {{config.FILE_INSPECTION.max_pdf_render_pages}}, "
        'DPI: {{config.FILE_INSPECTION.pdf_render_dpi}}; {{!other-skill}}; {{output-id}}; {"example":1}'
    )
    assert MarkdownLoader(markdown_path).load() == (
        'Pages: 2, DPI: 72; {{!other-skill}}; {{output-id}}; {"example":1}'
    )


@pytest.mark.parametrize(
    "reference",
    ["UNKNOWN.value", "FILE_INSPECTION.missing", "FILE_INSPECTION.image_extensions"],
)
def test_invalid_configuration_reference_fails_before_model_use(tmp_path, reference):
    markdown_path = tmp_path / "skill.md"
    markdown_path.write_text("{{config." + reference + "}}")
    with pytest.raises(ValueError, match="configuration reference"):
        MarkdownLoader(markdown_path).load()
