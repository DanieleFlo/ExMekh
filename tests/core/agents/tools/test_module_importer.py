import sys
from pathlib import Path
from typing import cast

import pytest

from qs.core.agents.tools import module_importer
from qs.core.agents.tools.module_importer import ToolModuleImporter

_DYNAMIC_PACKAGE_PREFIX = "_qs_deep_tool_"


def _dynamic_tool_modules() -> set[str]:
    return {
        module_name
        for module_name in sys.modules
        if module_name.startswith(_DYNAMIC_PACKAGE_PREFIX)
    }


def _write_entrypoint(
    tmp_path: Path, content: str = "VALUE = 1\n"
) -> tuple[Path, Path]:
    tool_path = tmp_path / "tool"
    tool_path.mkdir()
    entrypoint_path = tool_path / "main.py"
    entrypoint_path.write_text(content)
    return tool_path, entrypoint_path


def test_rejects_non_path_arguments(tmp_path: Path) -> None:
    importer = ToolModuleImporter()

    with pytest.raises(TypeError, match="tool_path must be a Path"):
        importer.import_module(cast(Path, "tool"), tmp_path / "main.py")

    with pytest.raises(TypeError, match="entrypoint_path must be a Path"):
        importer.import_module(tmp_path, cast(Path, "main.py"))


def test_rejects_entrypoint_outside_tool_folder(tmp_path: Path) -> None:
    tool_path = tmp_path / "tool"
    tool_path.mkdir()
    entrypoint_path = tmp_path / "main.py"
    entrypoint_path.write_text("VALUE = 1\n")

    with pytest.raises(
        ValueError,
        match="entrypoint_path must be inside tool_path",
    ):
        ToolModuleImporter().import_module(tool_path, entrypoint_path)


def test_rejects_non_path_when_unloading() -> None:
    with pytest.raises(TypeError, match="tool_path must be a Path"):
        ToolModuleImporter().unload_module(cast(Path, "tool"))


def test_rejects_symbolic_link_tool_path(tmp_path: Path) -> None:
    target_path, entrypoint_path = _write_entrypoint(tmp_path)
    tool_path = tmp_path / "link"
    tool_path.symlink_to(target_path, target_is_directory=True)

    with pytest.raises(ValueError, match="tool_path must not be a symbolic link"):
        ToolModuleImporter().import_module(tool_path, entrypoint_path)


def test_rejects_symbolic_link_entrypoint(tmp_path: Path) -> None:
    tool_path = tmp_path / "tool"
    tool_path.mkdir()
    target_path = tool_path / "target.py"
    target_path.write_text("VALUE = 1\n")
    entrypoint_path = tool_path / "main.py"
    entrypoint_path.symlink_to(target_path)

    with pytest.raises(
        ValueError,
        match="entrypoint_path must not be a symbolic link",
    ):
        ToolModuleImporter().import_module(tool_path, entrypoint_path)


def test_rejects_tool_path_that_is_not_directory(tmp_path: Path) -> None:
    tool_path = tmp_path / "tool"
    entrypoint_path = tmp_path / "main.py"

    with pytest.raises(ValueError, match="tool_path must be an existing directory"):
        ToolModuleImporter().import_module(tool_path, entrypoint_path)


def test_rejects_entrypoint_that_is_not_file(tmp_path: Path) -> None:
    tool_path = tmp_path / "tool"
    tool_path.mkdir()

    with pytest.raises(ValueError, match="entrypoint_path must be an existing file"):
        ToolModuleImporter().import_module(tool_path, tool_path / "main.py")


def test_removes_package_when_import_specification_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool_path, entrypoint_path = _write_entrypoint(tmp_path)
    modules_before_import = _dynamic_tool_modules()
    monkeypatch.setattr(
        module_importer,
        "spec_from_file_location",
        lambda *args: None,
    )

    with pytest.raises(ImportError, match="Cannot import tool entrypoint"):
        ToolModuleImporter().import_module(tool_path, entrypoint_path)

    assert _dynamic_tool_modules() == modules_before_import


def test_removes_package_when_module_execution_fails(tmp_path: Path) -> None:
    tool_path, entrypoint_path = _write_entrypoint(
        tmp_path,
        "raise RuntimeError('import failed')\n",
    )
    modules_before_import = _dynamic_tool_modules()

    with pytest.raises(RuntimeError, match="import failed"):
        ToolModuleImporter().import_module(tool_path, entrypoint_path)

    assert _dynamic_tool_modules() == modules_before_import
