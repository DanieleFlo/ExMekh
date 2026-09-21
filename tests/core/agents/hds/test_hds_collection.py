from pathlib import Path

import pytest

from qs.core.agents.hds.hds_collection import HDSCollection


def test_loads_hds_markdown_files_in_name_order(tmp_path: Path) -> None:
    folder_path = tmp_path / "hds"
    folder_path.mkdir()
    markdown = "---\ndescription: Instructions\nexit-tool: finish\n---\nBody"
    (folder_path / "second.md").write_text(markdown)
    (folder_path / "first.MD").write_text(markdown)

    resources = HDSCollection([folder_path]).load()

    assert tuple(resource.name for resource in resources) == ("first", "second")


def test_rejects_duplicate_hds_folder_paths(tmp_path: Path) -> None:
    folder_path = tmp_path / "hds"

    with pytest.raises(ValueError, match="Duplicate HDS folder path"):
        HDSCollection([folder_path, folder_path])
