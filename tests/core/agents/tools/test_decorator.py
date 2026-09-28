from collections.abc import Callable
from pathlib import Path
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


@pytest.mark.parametrize("background_enabled", [None, 0, "false"])
def test_rejects_non_boolean_background_policy(background_enabled) -> None:
    with pytest.raises(TypeError, match="background_enabled must be a bool"):
        deep_tool(background_enabled=background_enabled)


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
    def entrypoint(work_root: Path) -> Path:
        return work_root

    with pytest.raises(ValueError, match="already decorated"):
        decorator(entrypoint)


def test_passes_work_root_to_decorated_function(tmp_path: Path) -> None:
    @deep_tool(description="Description.")
    def entrypoint(work_root: Path, value: str) -> tuple[Path, str]:
        return work_root, value

    assert entrypoint(tmp_path, "value") == (tmp_path, "value")


@pytest.mark.parametrize(
    "entrypoint",
    [
        lambda: None,
        lambda root: None,
        lambda *, work_root: None,
    ],
)
def test_requires_work_root_as_first_positional_parameter(
    entrypoint: Callable[..., None],
) -> None:
    with pytest.raises(TypeError, match="work_root as its first positional parameter"):
        deep_tool(description="Description.")(entrypoint)
