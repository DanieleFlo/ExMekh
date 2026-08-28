from typing import cast

import pytest

from qs.core.agents.exit_tool.decorator import (
    ExitToolDefinition,
    exit_tool,
    get_exit_tool_definition,
)


def test_decorates_function_and_normalizes_description() -> None:
    @exit_tool(name="finish", description=" Finish work ")
    def finish() -> None:
        pass

    assert get_exit_tool_definition(finish) == ExitToolDefinition(
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
        exit_tool(name=name, description=description)


def test_rejects_non_function_and_duplicate_decoration() -> None:
    decorator = exit_tool()

    with pytest.raises(TypeError, match="can decorate only functions"):
        decorator(cast(object, lambda: None).__class__)

    @decorator
    def finish() -> None:
        pass

    with pytest.raises(ValueError, match="already decorated"):
        decorator(finish)
