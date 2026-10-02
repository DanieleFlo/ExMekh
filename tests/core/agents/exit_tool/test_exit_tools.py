import sys
from pathlib import Path

import pytest

from exmekh_lib.core.agents.exit_tool.exit_tools import ExitTools


def _dynamic_tool_modules() -> set[str]:
    return {name for name in sys.modules if name.startswith("_qs_deep_tool_")}


def _write_tool(path: Path, body: str) -> Path:
    path.mkdir(parents=True)
    (path / "main.py").write_text(body)
    (path / "instructions.md").write_text("Finish now")
    return path


def _valid_source(*, description: str = "'Exit agent'") -> str:
    return (
        "from exmekh_lib.core.agents.exit_tool.data import ExitToolData\n"
        "from exmekh_lib.core.agents.exit_tool.decorator import exit_tool\n\n"
        f"@exit_tool(instructions_path='instructions.md', name='exit_agent', description={description})\n"
        "def finish(text: str, items: list[str]) -> ExitToolData:\n"
        "    return ExitToolData(text=text, items=items)\n"
    )


def test_loads_exit_tool_and_rejects_second_load(tmp_path: Path) -> None:
    tool_path = _write_tool(tmp_path / "finish", _valid_source())
    loader = ExitTools([tool_path])

    (tool,) = loader.load_exit_tools()

    assert tool.name == "exit_agent"
    assert tool.description == "Exit agent"
    assert tool.instructions == "Finish now"
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
            + "\n@exit_tool(instructions_path='instructions.md', description='Other')\ndef other() -> ExitToolData:\n    return ExitToolData('x', ['y'])\n",
            "exactly one @exit_tool entrypoint",
        ),
        (
            (
                "from exmekh_lib.core.agents.exit_tool.decorator import exit_tool\n"
                "@exit_tool(instructions_path='instructions.md', description='Exit')\n"
                "def finish() -> str:\n    return 'wrong'\n"
            ),
            "must return ExitToolData",
        ),
        (
            (
                "from __future__ import annotations\n"
                "from exmekh_lib.core.agents.exit_tool.decorator import exit_tool\n"
                "@exit_tool(instructions_path='instructions.md', description='Exit')\n"
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


@pytest.mark.parametrize(
    ("filename", "content", "error", "message"),
    [
        ("missing.md", None, FileNotFoundError, "does not exist"),
        ("empty.md", "", ValueError, "cannot be blank"),
        ("blank.md", " \n\t", ValueError, "cannot be blank"),
        ("instructions.txt", "Finish", ValueError, "must be a Markdown file"),
        ("directory.md", None, ValueError, "must be a file"),
    ],
)
def test_rejects_invalid_instructions_and_unloads_module(
    tmp_path, filename, content, error, message
):
    path = _write_tool(
        tmp_path / "finish", _valid_source().replace("instructions.md", filename)
    )
    if content is not None:
        (path / filename).write_text(content)
    elif filename == "directory.md":
        (path / filename).mkdir()
    before = _dynamic_tool_modules()
    with pytest.raises(error, match=message):
        ExitTools([path]).load_exit_tools()
    assert _dynamic_tool_modules() == before


@pytest.mark.parametrize("absolute", [False, True])
def test_instruction_paths_are_independent_of_working_directory(
    tmp_path, monkeypatch, absolute
):
    path = _write_tool(tmp_path / "finish", _valid_source())
    instructions = path / "instructions.md"
    instructions.write_text(
        "  Finish with {{!exit_agent}} e {{ !other-tool }}; {{!exit_agent}}.  ",
        encoding="utf-8",
    )
    if absolute:
        source = (path / "main.py").read_text()
        (path / "main.py").write_text(
            source.replace("'instructions.md'", repr(str(instructions)))
        )
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    (unrelated / "instructions.md").write_text("WRONG FILE")
    monkeypatch.chdir(unrelated)
    loader = ExitTools([path])
    try:
        (tool,) = loader.load_exit_tools()
        assert tool.instructions_path == instructions
        assert tool.references == ("exit_agent", "other-tool")
        assert tool.instructions == "Finish with exit_agent e other-tool; exit_agent."
        instructions.unlink()
        assert "Finish" in tool.render_prompt_section()
        assert tool.execute(text="done", items=["result"]) == {
            "text": "done",
            "items": ["result"],
        }
    finally:
        loader.unload()


def test_instructions_are_plain_markdown_without_yaml_parsing(tmp_path):
    path = _write_tool(tmp_path / "finish", _valid_source())
    text = "---\nnot: [valid YAML\n---\n# Finish\nUse {{!exit_agent}}"
    (path / "instructions.md").write_text(text)
    loader = ExitTools([path])
    try:
        (tool,) = loader.load_exit_tools()
        assert tool.instructions == text.replace("{{!exit_agent}}", "exit_agent")
    finally:
        loader.unload()


def test_rejects_symlink_instruction_file(tmp_path):
    path = _write_tool(tmp_path / "finish", _valid_source())
    target = tmp_path / "target.md"
    target.write_text("Finish")
    instructions = path / "instructions.md"
    instructions.unlink()
    instructions.symlink_to(target)
    with pytest.raises(ValueError, match="must not be a symbolic link"):
        ExitTools([path]).load_exit_tools()


def test_instruction_failure_unloads_previously_loaded_tools(tmp_path):
    first = _write_tool(tmp_path / "first", _valid_source())
    second = _write_tool(tmp_path / "second", _valid_source())
    (second / "instructions.md").write_text(" ")
    before = _dynamic_tool_modules()
    with pytest.raises(ValueError, match="cannot be blank"):
        ExitTools([first, second]).load_exit_tools()
    assert _dynamic_tool_modules() == before


def test_invalid_utf8_instructions_abort_loading(tmp_path):
    path = _write_tool(tmp_path / "finish", _valid_source())
    (path / "instructions.md").write_bytes(b"\xff")
    with pytest.raises(OSError, match="reading the file"):
        ExitTools([path]).load_exit_tools()
