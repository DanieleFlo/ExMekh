from pathlib import Path

import pytest

from qs.core.agents.exit_tool.exit_tools import ExitTools


def _write_tool(path: Path, body: str) -> Path:
    path.mkdir(parents=True)
    (path / "main.py").write_text(body)
    return path


def _valid_source(*, description: str = "'Exit agent'") -> str:
    return (
        "from qs.core.agents.exit_tool.data import ExitToolData\n"
        "from qs.core.agents.exit_tool.decorator import exit_tool\n\n"
        f"@exit_tool(name='exit_agent', description={description})\n"
        "def finish(text: str, items: list[str]) -> ExitToolData:\n"
        "    return ExitToolData(text=text, items=items)\n"
    )


def test_loads_exit_tool_and_rejects_second_load(tmp_path: Path) -> None:
    tool_path = _write_tool(tmp_path / "finish", _valid_source())
    loader = ExitTools([tool_path])

    (tool,) = loader.load_exit_tools()

    assert tool.name == "exit_agent"
    assert tool.description == "Exit agent"
    assert tool.exit_skill_name == "finish"
    assert tool.folder_path == tool_path
    with pytest.raises(RuntimeError, match="load only once"):
        loader.load_exit_tools()


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (
            "def finish() -> None:\n    pass\n",
            "exactly one @exit_tool entrypoint",
        ),
        (
            _valid_source()
            + "\n@exit_tool(description='Other')\ndef other() -> ExitToolData:\n    return ExitToolData('x', ['y'])\n",
            "exactly one @exit_tool entrypoint",
        ),
        (
            (
                "from qs.core.agents.exit_tool.decorator import exit_tool\n"
                "@exit_tool(description='Exit')\n"
                "def finish() -> str:\n    return 'wrong'\n"
            ),
            "must return ExitToolData",
        ),
        (
            (
                "from __future__ import annotations\n"
                "from qs.core.agents.exit_tool.decorator import exit_tool\n"
                "@exit_tool(description='Exit')\n"
                "def finish() -> MissingType:\n    return None\n"
            ),
            "must return ExitToolData",
        ),
        (_valid_source(description="None"), "must have a description"),
    ],
)
def test_rejects_invalid_exit_tool_modules(
    tmp_path: Path,
    source: str,
    message: str,
) -> None:
    tool_path = _write_tool(tmp_path / "finish", source)

    with pytest.raises(ValueError, match=message):
        ExitTools([tool_path]).load_exit_tools()


def test_unloads_previously_loaded_tools_when_later_tool_fails(tmp_path: Path) -> None:
    valid = _write_tool(tmp_path / "first", _valid_source())
    invalid = _write_tool(tmp_path / "second", "def undecorated():\n    pass\n")
    loader = ExitTools([valid, invalid])

    with pytest.raises(ValueError, match="exactly one"):
        loader.load_exit_tools()

    loader.unload()
