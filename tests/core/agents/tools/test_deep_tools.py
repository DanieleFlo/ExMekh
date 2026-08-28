import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import pytest

from qs.core.agents.tools.deep_tools import DeepTools

_DYNAMIC_PACKAGE_PREFIX = "_qs_deep_tool_"


def _dynamic_tool_modules() -> set[str]:
    return {
        module_name
        for module_name in sys.modules
        if module_name.startswith(_DYNAMIC_PACKAGE_PREFIX)
    }


def _write_tool(
    tmp_path: Path,
    *,
    folder_name: str = "search",
    main: str,
    files: dict[str, str] | None = None,
) -> Path:
    tool_path = tmp_path / folder_name
    tool_path.mkdir()
    (tool_path / "main.py").write_text(main)

    for filename, content in (files or {}).items():
        (tool_path / filename).write_text(content)

    return tool_path


def test_loads_decorated_entrypoint_with_metadata_and_relative_imports(
    tmp_path: Path,
) -> None:
    tool_path = _write_tool(
        tmp_path,
        main=(
            "from qs.core.agents.tools.decorator import deep_tool\n"
            "\n"
            "@deep_tool(name='lookup', description='Search the index.')\n"
            "def run(query: str) -> str:\n"
            "    from .helper import PREFIX\n"
            "    return f'{PREFIX}: {query}'\n"
        ),
        files={"helper.py": "PREFIX = 'result'\n"},
    )

    tools = DeepTools([tool_path]).load()

    assert len(tools) == 1
    assert tools[0].name == "lookup"
    assert tools[0].description == "Search the index."
    assert tools[0].tool("documentation") == "result: documentation"


def test_uses_folder_name_as_default(tmp_path: Path) -> None:
    tool_path = _write_tool(
        tmp_path,
        folder_name="file-search",
        main=(
            "from qs.core.agents.tools.decorator import deep_tool\n\n"
            "@deep_tool(description='Search project files.')\n"
            "def run() -> str:\n"
            "    return 'result'\n"
        ),
    )

    tool = DeepTools([tool_path]).load()[0]

    assert tool.name == "file-search"
    assert tool.description == "Search project files."


def test_rejects_missing_description_even_with_docstring(tmp_path: Path) -> None:
    tool_path = _write_tool(
        tmp_path,
        main=(
            "from qs.core.agents.tools.decorator import deep_tool\n\n"
            "@deep_tool()\n"
            "def run() -> str:\n"
            '    """Search project files."""\n'
            "    return 'result'\n"
        ),
    )

    with pytest.raises(ValueError, match="must have a description"):
        DeepTools([tool_path]).load()


def test_rejects_string_as_tool_path_collection() -> None:
    with pytest.raises(TypeError, match="tool_paths must be a sequence of Path"):
        DeepTools(cast(Sequence[Path], "search"))


def test_rejects_non_path_collection_item() -> None:
    with pytest.raises(TypeError, match="tool_paths must contain only Path"):
        DeepTools([cast(Path, "search")])


def test_rejects_duplicate_tool_paths(tmp_path: Path) -> None:
    tool_path = tmp_path / "search"

    with pytest.raises(ValueError, match="Duplicate tool path"):
        DeepTools([tool_path, tool_path])


def test_rejects_missing_tool_folder(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Tool path does not exist"):
        DeepTools([tmp_path / "search"]).load()


def test_rejects_tool_path_that_is_not_directory(tmp_path: Path) -> None:
    tool_path = tmp_path / "search"
    tool_path.write_text("not a directory")

    with pytest.raises(ValueError, match="Tool path must be a directory"):
        DeepTools([tool_path]).load()


def test_rejects_symbolic_link_tool_folder(tmp_path: Path) -> None:
    target_path = tmp_path / "target"
    target_path.mkdir()
    tool_path = tmp_path / "search"
    tool_path.symlink_to(target_path, target_is_directory=True)

    with pytest.raises(ValueError, match="must not be a symbolic link"):
        DeepTools([tool_path]).load()


def test_rejects_tool_folder_without_main(tmp_path: Path) -> None:
    tool_path = tmp_path / "search"
    tool_path.mkdir()

    with pytest.raises(
        FileNotFoundError,
        match="Tool entrypoint does not exist",
    ):
        DeepTools([tool_path]).load()


def test_rejects_symlink_entrypoint(tmp_path: Path) -> None:
    target_path = tmp_path / "external.py"
    target_path.write_text("def run() -> None:\n    pass\n")
    tool_path = tmp_path / "search"
    tool_path.mkdir()
    (tool_path / "main.py").symlink_to(target_path)

    with pytest.raises(ValueError, match="must not be a symbolic link"):
        DeepTools([tool_path]).load()


def test_rejects_entrypoint_that_is_not_file(tmp_path: Path) -> None:
    tool_path = tmp_path / "search"
    (tool_path / "main.py").mkdir(parents=True)

    with pytest.raises(ValueError, match="entrypoint must be a file"):
        DeepTools([tool_path]).load()


def test_rejects_main_without_decorated_entrypoint(tmp_path: Path) -> None:
    tool_path = _write_tool(
        tmp_path,
        main="def run() -> None:\n    pass\n",
    )

    with pytest.raises(
        ValueError,
        match="must define exactly one @deep_tool entrypoint",
    ):
        DeepTools([tool_path]).load()


def test_removes_dynamic_modules_when_entrypoint_is_invalid(
    tmp_path: Path,
) -> None:
    tool_path = _write_tool(
        tmp_path,
        main="def run() -> None:\n    pass\n",
    )
    modules_before_load = _dynamic_tool_modules()

    with pytest.raises(ValueError):
        DeepTools([tool_path]).load()

    assert _dynamic_tool_modules() == modules_before_load


def test_rolls_back_previously_loaded_tools_when_batch_fails(
    tmp_path: Path,
) -> None:
    valid_tool_path = _write_tool(
        tmp_path,
        folder_name="valid",
        main=(
            "from qs.core.agents.tools.decorator import deep_tool\n\n"
            "@deep_tool(description='Valid tool.')\n"
            "def run() -> None:\n"
            "    pass\n"
        ),
    )
    invalid_tool_path = _write_tool(
        tmp_path,
        folder_name="invalid",
        main="def run() -> None:\n    pass\n",
    )
    modules_before_load = _dynamic_tool_modules()

    with pytest.raises(ValueError):
        DeepTools([valid_tool_path, invalid_tool_path]).load()

    assert _dynamic_tool_modules() == modules_before_load


def test_independent_loaders_for_same_path_are_isolated(
    tmp_path: Path,
) -> None:
    tool_path = _write_tool(
        tmp_path,
        main=(
            "from qs.core.agents.tools.decorator import deep_tool\n\n"
            "@deep_tool(description='Lazy tool.')\n"
            "def run() -> str:\n"
            "    from .helper import result\n"
            "    return result()\n"
        ),
    )
    (tool_path / "helper.py").write_text("def result() -> str:\n    return 'result'\n")
    first_loader = DeepTools([tool_path])
    second_loader = DeepTools([tool_path])
    first_tool = first_loader.load()[0]
    second_loader.load()

    second_loader.unload()

    assert first_tool.tool() == "result"
    first_loader.unload()


def test_loader_cannot_load_more_than_once(tmp_path: Path) -> None:
    tool_path = _write_tool(
        tmp_path,
        main=(
            "from qs.core.agents.tools.decorator import deep_tool\n\n"
            "@deep_tool(description='Tool.')\n"
            "def run() -> None:\n"
            "    pass\n"
        ),
    )
    loader = DeepTools([tool_path])
    loader.load()

    with pytest.raises(RuntimeError, match="can load only once"):
        loader.load()

    loader.unload()


def test_rejects_multiple_decorated_entrypoints(tmp_path: Path) -> None:
    tool_path = _write_tool(
        tmp_path,
        main=(
            "from qs.core.agents.tools.decorator import deep_tool\n\n"
            "@deep_tool()\n"
            "def first() -> None:\n"
            "    pass\n\n"
            "@deep_tool()\n"
            "def second() -> None:\n"
            "    pass\n"
        ),
    )

    with pytest.raises(
        ValueError,
        match="must define exactly one @deep_tool entrypoint",
    ):
        DeepTools([tool_path]).load()
