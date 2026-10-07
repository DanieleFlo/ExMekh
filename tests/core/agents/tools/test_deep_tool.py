from pathlib import Path

import pytest

from exmekh_lib.core.agents.tools.decorator import deep_tool
from exmekh_lib.core.agents.tools.deep_tool import DeepTool


def test_binds_runtime_dependencies_without_exposing_them_to_the_model(
    tmp_path: Path,
) -> None:
    @deep_tool(name="lookup", description="Lookup a value.", background_enabled=False)
    def entrypoint(work_root: Path, prefix: str, value: str) -> str:
        assert work_root == tmp_path
        return prefix + value

    tool = DeepTool(entrypoint, tmp_path, "bound:", cache_result=False)

    assert tool.execute(value="value") == "bound:value"
    assert tuple(tool.get_signature().parameters) == ("value",)
    assert tool.get_name() == "lookup"
    assert tool.get_description() == "Lookup a value."
    assert tool.cache_result is False


def test_accepts_explicit_name_and_description_overrides(tmp_path: Path) -> None:
    @deep_tool(name="original", description="Original description.")
    def entrypoint(work_root: Path) -> None:
        del work_root

    original = DeepTool(entrypoint, tmp_path)
    renamed = DeepTool(
        entrypoint, tmp_path, name="renamed", description="Replacement description."
    )

    assert original.get_name() == "original"
    assert original.get_description() == "Original description."
    assert renamed.get_name() == "renamed"
    assert renamed.get_description() == "Replacement description."


@pytest.mark.parametrize("decorate", [False, True])
def test_requires_a_decorated_entrypoint_with_a_description(
    tmp_path: Path, decorate: bool
) -> None:
    def entrypoint(work_root: Path) -> None:
        del work_root

    if decorate:
        entrypoint = deep_tool()(entrypoint)

    with pytest.raises(ValueError, match="@deep_tool with a description"):
        DeepTool(entrypoint, tmp_path)


def test_rejects_an_invalid_name_override(tmp_path: Path) -> None:
    @deep_tool(name="valid", description="Description.")
    def entrypoint(work_root: Path) -> None:
        del work_root

    with pytest.raises(ValueError, match="Tool name must contain"):
        DeepTool(entrypoint, tmp_path, name="")


def test_argument_guidance_is_bound_without_becoming_a_model_argument(tmp_path):
    @deep_tool(description="Read content", argument_guidance="Inspect first.")
    def entrypoint(work_root: Path, path: str) -> str:
        return path

    tool = DeepTool(entrypoint, tmp_path)
    assert "argument_guidance" not in tool.get_signature().parameters
    with pytest.raises(TypeError, match="Inspect first"):
        tool.execute_with_hooks({}, work_root=tmp_path, history=())
