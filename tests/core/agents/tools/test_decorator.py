from collections.abc import Callable
from typing import Any, cast

import pytest

from qs.core.agents.tools.decorator import deep_tool


@pytest.mark.parametrize(
    ("name", "description", "message"),
    [
        (cast(str, 42), "Description.", "name must be a string or None"),
        ("search", cast(str, 42), "description must be a string or None"),
        ("search", "   ", "description cannot be blank"),
    ],
)
def test_rejects_invalid_metadata(
    name: str | None,
    description: str | None,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        deep_tool(name=name, description=description)


def test_rejects_non_function_entrypoint() -> None:
    class CallableObject:
        def __call__(self) -> None:
            pass

    decorator = deep_tool(description="Description.")

    with pytest.raises(TypeError, match="can decorate only functions"):
        decorator(cast(Callable[..., Any], CallableObject()))


def test_rejects_multiple_decorators_on_same_function() -> None:
    decorator = deep_tool(description="Description.")

    @decorator
    def entrypoint() -> None:
        pass

    with pytest.raises(ValueError, match="already decorated"):
        decorator(entrypoint)
