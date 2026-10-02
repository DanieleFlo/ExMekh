from pathlib import Path
from typing import cast

import pytest

from exmekh_lib.core.agents.deep_skill.deep_skills import DeepSkills


def _exit_instructions() -> str:
    return '<exit-tool name="finish">Finish instructions</exit-tool>'


def test_rejects_string_as_folder_collection() -> None:
    with pytest.raises(
        TypeError,
        match="folder_paths must be a sequence of Path",
    ):
        DeepSkills(cast(list[Path], "deep-skill"))


def test_rejects_non_path_collection_item() -> None:
    with pytest.raises(
        TypeError,
        match="folder_paths must contain only Path",
    ):
        DeepSkills(cast(list[Path], ["deep-skill"]))


def test_rejects_duplicate_folder_paths(tmp_path: Path) -> None:
    folder_path = tmp_path / "deep-skill"

    with pytest.raises(ValueError, match="Duplicate deep-skill folder path"):
        DeepSkills([folder_path, folder_path])


def test_rejects_symbolic_link_folder(tmp_path: Path) -> None:
    target_path = tmp_path / "target"
    target_path.mkdir()
    folder_path = tmp_path / "deep-skill"
    folder_path.symlink_to(target_path, target_is_directory=True)

    with pytest.raises(ValueError, match="must not be a symbolic link"):
        DeepSkills([folder_path]).load()


def test_rejects_symbolic_link_markdown(tmp_path: Path) -> None:
    folder_path = tmp_path / "deep-skill"
    folder_path.mkdir()
    target_path = tmp_path / "skill.md"
    target_path.write_text("content")
    (folder_path / "skill.md").symlink_to(target_path)

    with pytest.raises(ValueError, match="must not be a symbolic link"):
        DeepSkills([folder_path]).load()


def test_rejects_missing_folder(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="folder does not exist"):
        DeepSkills([tmp_path / "deep-skill"]).load()


def test_rejects_folder_path_that_is_not_directory(tmp_path: Path) -> None:
    folder_path = tmp_path / "deep-skill"
    folder_path.write_text("not a directory")

    with pytest.raises(ValueError, match="path must be a directory"):
        DeepSkills([folder_path]).load()


def test_rejects_nested_directory(tmp_path: Path) -> None:
    folder_path = tmp_path / "deep-skill"
    (folder_path / "nested").mkdir(parents=True)

    with pytest.raises(ValueError, match="only Markdown files"):
        DeepSkills([folder_path]).load()


def test_rejects_non_markdown_entry(tmp_path: Path) -> None:
    folder_path = tmp_path / "deep-skill"
    folder_path.mkdir()
    (folder_path / "skill.txt").write_text("content")

    with pytest.raises(ValueError, match="must have '.md' extension"):
        DeepSkills([folder_path]).load()


def test_loads_markdown_files_in_name_order(tmp_path: Path) -> None:
    folder_path = tmp_path / "deep-skill"
    folder_path.mkdir()
    markdown = "---\ndescription: Description\n---\nBody"
    (folder_path / "second.md").write_text(markdown)
    (folder_path / "first.MD").write_text(markdown)

    skills = DeepSkills([folder_path]).load()

    assert tuple(skill.name for skill in skills) == ("first", "second")
    assert skills[0].render_instructions(_exit_instructions()) == (
        'Body\n\n<exit-tool name="finish">Finish instructions</exit-tool>'
    )
    assert skills[0].catalog is None
