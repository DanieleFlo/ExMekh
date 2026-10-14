import inspect
from pathlib import Path
from typing import cast

import pytest

from exmekh_lib.core.models.assistant_message import (
    FunctionCall,
    ToolCall,
)
from exmekh_lib.core.models.interactive import InteractiveDefinition
from exmekh_lib.core.models.tool import (
    AttachedToolResult,
    CalledTool,
    ExecutionOptions,
    PreformattedToolResult,
    Tool,
)
from exmekh_lib.core.models.user_message import UserMessage
from exmekh_lib.utils.tool_errors import ToolArgumentError


def test_binding_error_does_not_claim_a_present_positional_only_name_is_missing():
    def work(value, /, **options):
        pytest.fail("Invalid keyword binding must not execute work")

    with pytest.raises(ToolArgumentError) as caught:
        Tool(work).validate_arguments({"value": "provided as a keyword"})
    assert caught.value.code == "invalid_tool_arguments"
    assert caught.value.details["missing"] == []
    assert caught.value.details["unexpected"] == []


def documented_tool(query: str) -> str:
    """Search the documentation."""
    return query


def test_tool_uses_callable_metadata_by_default() -> None:
    tool = Tool(documented_tool)

    assert tool.tool is documented_tool
    assert tool.name == "documented_tool"
    assert tool.description == "Search the documentation."


def test_tool_accepts_explicit_metadata() -> None:
    tool = Tool(documented_tool, name="search-docs", description="Search docs")

    assert tool.name == "search-docs"
    assert tool.description == "Search docs"
    assert tool.get_name() == "search-docs"
    assert tool.get_description() == "Search docs"
    assert tuple(tool.get_signature().parameters) == (
        "query",
        "execution_mode",
        "wait_seconds",
    )
    assert tool.get_type_hints()["query"] is str
    assert tool.get_type_hints()["return"] is str
    assert tool.cache_result is True


def test_tool_creates_copy_with_description() -> None:
    tool = Tool(documented_tool, name="search-docs", description="Search docs")

    described_tool = tool.with_description("Available in skill 'docs'.")

    assert described_tool is not tool
    assert described_tool.tool is documented_tool
    assert described_tool.name == "search-docs"
    assert described_tool.description == "Available in skill 'docs'."
    assert described_tool.cache_result is True
    assert tool.description == "Search docs"


def test_tool_creates_copy_with_description_suffix() -> None:
    tool = Tool(documented_tool, name="search-docs", description="Search docs.")

    described_tool = tool.with_description_suffix("Available in every skill.")

    assert described_tool.description == "Search docs. Available in every skill."


@pytest.mark.parametrize("original_mode", ["background", "synchronous", "interactive"])
@pytest.mark.parametrize("replacement_background", [False, True])
def test_execution_policy_copies_keep_contracts_without_enabling_background(
    original_mode, replacement_background
):
    class SpecializedTool(Tool):
        pass

    definition = (
        InteractiveDefinition("edit_file.v3", {"document": "query"})
        if original_mode == "interactive"
        else None
    )
    original = Tool(
        documented_tool,
        background_enabled=original_mode == "background",
        interactive=definition,
    )
    replacement = SpecializedTool(
        documented_tool,
        name="replacement",
        description="Replacement",
        background_enabled=replacement_background,
        cache_result=False,
    )
    composed = replacement.with_execution_policy_of(original)
    assert isinstance(composed, SpecializedTool)
    assert composed is not replacement
    assert composed.get_name() == "replacement"
    assert composed.get_description() == "Replacement"
    assert composed.execute(query="value") == "value"
    assert not composed.cache_result
    assert composed.interactive is definition
    assert composed.background_enabled == (
        original_mode == "background" and replacement_background
    )
    assert replacement.background_enabled == replacement_background
    assert replacement.interactive is None
    assert original.background_enabled == (original_mode == "background")


@pytest.mark.parametrize("name", ["", "contains spaces", "x" * 65])
def test_tool_rejects_invalid_openai_name(name: str) -> None:
    with pytest.raises(ValueError, match="Tool name"):
        Tool(documented_tool, name=name)


def test_tool_rejects_non_callable() -> None:
    with pytest.raises(TypeError, match="callable"):
        Tool("not-callable")


def test_tool_rejects_non_string_description() -> None:
    with pytest.raises(TypeError, match="description must be a string"):
        Tool(documented_tool, description=cast(str, object()))


def test_tool_argument_guidance_is_validated_and_preserved_in_copies():
    with pytest.raises(TypeError, match="argument_guidance must be a string"):
        Tool(documented_tool, argument_guidance=None)
    tool = Tool(documented_tool, argument_guidance="Use query='topic'.")
    for copied in (tool.with_name("renamed"), tool.with_description("contextual")):
        with pytest.raises(TypeError, match="Use query='topic'"):
            copied.execute_with_hooks({}, work_root=Path.cwd(), history=())


def test_explicit_empty_description_matches_description_copies() -> None:
    tool = Tool(documented_tool, description="")
    assert tool.get_description() == tool.with_description("").get_description() == ""


@pytest.mark.parametrize("description", [False, 0, []])
def test_tool_rejects_falsey_non_string_descriptions(description) -> None:
    with pytest.raises(TypeError, match="description must be a string"):
        Tool(documented_tool, description=description)


def test_tool_preserves_disabled_cache_policy_in_description_copies() -> None:
    tool = Tool(documented_tool, cache_result=False)

    assert tool.with_description("Runtime tool").cache_result is False
    assert tool.with_description_suffix("Always available.").cache_result is False


def test_tool_rejects_non_boolean_cache_policy() -> None:
    with pytest.raises(TypeError, match="cache_result must be a bool"):
        Tool(documented_tool, cache_result=cast(bool, "false"))


@pytest.mark.parametrize("background_enabled", [None, 0, "false"])
def test_tool_rejects_non_boolean_background_policy(background_enabled) -> None:
    with pytest.raises(TypeError, match="background_enabled must be a bool"):
        Tool(documented_tool, background_enabled=background_enabled)


@pytest.mark.parametrize(
    "function",
    [lambda execution_mode: execution_mode, lambda wait_seconds: wait_seconds],
)
def test_reserved_execution_parameters_require_synchronous_tools(function) -> None:
    with pytest.raises(ValueError, match="reserved execution names"):
        Tool(function, name="work")

    tool = Tool(function, name="work", background_enabled=False)
    parameter_name = next(iter(inspect.signature(function).parameters))
    assert tool.execute(**{parameter_name: "business value"}) == "business value"


def test_execution_options_reject_unknown_modes() -> None:
    with pytest.raises(ValueError, match="execution_mode must be"):
        ExecutionOptions(mode="unknown")


def test_execution_parameters_precede_variadic_keyword_arguments() -> None:
    def work(value: str, **options: object) -> str:
        return value

    signature = Tool(work).get_signature()

    assert tuple(signature.parameters) == (
        "value",
        "execution_mode",
        "wait_seconds",
        "options",
    )
    assert signature.parameters["options"].kind == inspect.Parameter.VAR_KEYWORD
    assert signature.bind(
        value="text", execution_mode="background", extra=1
    ).arguments == {
        "value": "text",
        "execution_mode": "background",
        "options": {"extra": 1},
    }


def test_synchronous_tool_schema_exposes_only_business_parameters() -> None:
    tool = Tool(documented_tool, background_enabled=False)

    assert tool.get_type_hints() == {"query": str, "return": str}
    assert tuple(tool.get_signature().parameters) == ("query",)


def test_preformatted_tool_result_requires_text() -> None:
    assert PreformattedToolResult("<result />").content == "<result />"
    with pytest.raises(TypeError, match="content must be a string"):
        PreformattedToolResult(cast(str, object()))


def test_called_tool_executes_arguments_and_exposes_response_metadata() -> None:
    called_tool = CalledTool(
        Tool(documented_tool, name="search"),
        ToolCall(
            id="call-123",
            function=FunctionCall(
                name="search",
                arguments={"query": "clean code"},
            ),
        ),
    )

    assert called_tool.execute() == "clean code"
    assert called_tool.name == "search"
    assert called_tool.call_id == "call-123"


def test_called_tool_rejects_a_mismatched_name() -> None:
    with pytest.raises(ValueError, match="does not match"):
        CalledTool(
            Tool(documented_tool, name="search"),
            ToolCall(
                id="call-123",
                function=FunctionCall(name="other", arguments={}),
            ),
        )


def test_called_tool_rejects_invalid_types() -> None:
    tool = Tool(documented_tool)
    tool_call = ToolCall(
        id="call-123",
        function=FunctionCall(name="documented_tool", arguments={}),
    )

    with pytest.raises(TypeError, match="tool must be a Tool"):
        CalledTool(cast(Tool, object()), tool_call)
    with pytest.raises(TypeError, match="tool_call must be a ToolCall"):
        CalledTool.from_tool_call(tool, cast(ToolCall, object()))
    with pytest.raises(TypeError, match="tool_call must be a ToolCall"):
        CalledTool(tool, cast(ToolCall, object()))


def test_called_tool_factory_returns_valid_wrapper() -> None:
    tool = Tool(documented_tool)
    tool_call = ToolCall(
        id="call-123",
        function=FunctionCall(
            name="documented_tool",
            arguments={"query": "docs"},
        ),
    )

    assert CalledTool.from_tool_call(tool, tool_call).execute() == "docs"


def test_attached_tool_results_keep_companions_out_of_structured_fields():
    message = UserMessage("attachment")
    result = AttachedToolResult({"ok": True}, message)
    assert dict(result) == {"ok": True}
    assert result.attachments is message
    with pytest.raises(TypeError, match="require a UserMessage"):
        AttachedToolResult({"ok": True}, None)
