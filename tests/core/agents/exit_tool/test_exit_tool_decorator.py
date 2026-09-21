from pathlib import Path
from typing import cast

import pytest

from qs.core.agents.exit_tool.decorator import (
    ExitToolDefinition,
    exit_tool,
    get_exit_tool_definition,
)


def test_decorates_function_and_normalizes_description() -> None:
    @exit_tool(
        instructions_path="instructions.md", name="finish", description=" Finish work "
    )
    def finish() -> None:
        pass

    assert get_exit_tool_definition(finish) == ExitToolDefinition(
        instructions_path=Path("instructions.md"),
        name="finish",
        description="Finish work",
    )
    assert get_exit_tool_definition(object()) is None


@pytest.mark.parametrize(
    ("name", "description", "error_type", "message"),
    [
        (cast(str, 1), None, TypeError, "name must be a string or None"),
        (None, cast(str, 1), TypeError, "description must be a string or None"),
        (None, "   ", ValueError, "description cannot be blank"),
    ],
)
def test_rejects_invalid_definition(
    name: str | None,
    description: str | None,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        exit_tool(
            instructions_path="instructions.md", name=name, description=description
        )


def test_rejects_non_function_and_duplicate_decoration() -> None:
    decorator = exit_tool(instructions_path="instructions.md")

    with pytest.raises(TypeError, match="can decorate only functions"):
        decorator(cast(object, lambda: None).__class__)

    @decorator
    def finish() -> None:
        pass

    with pytest.raises(ValueError, match="already decorated"):
        decorator(finish)


@pytest.mark.parametrize(
    ("path", "error"),
    [
        (None, TypeError),
        (1, TypeError),
        ([], TypeError),
        ("", ValueError),
        (" \t", ValueError),
    ],
)
def test_rejects_invalid_instruction_path(path, error):
    with pytest.raises(error, match="instructions_path"):
        exit_tool(instructions_path=path)


def test_instruction_path_is_required():
    with pytest.raises(TypeError, match="instructions_path"):
        exit_tool()


def test_decorator_accepts_path_without_reading_file(tmp_path):
    path = tmp_path / "not-created-yet.md"

    @exit_tool(instructions_path=path)
    def finish():
        pass

    assert get_exit_tool_definition(finish).instructions_path == path
    assert not path.exists()
