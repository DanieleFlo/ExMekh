from functools import partial
from pathlib import Path
from typing import Annotated, Literal, TypedDict, cast

import pytest
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
)
from openai.types.chat.chat_completion_message_function_tool_call import (
    Function as OpenAIFunction,
)

from exmekh_lib.core.models.tool import Tool
from exmekh_lib.core.models.tools import Tools
from exmekh_lib.core.providers.openai_chat_completion_tool_mapper import (
    OpenAIChatCompletionToolMapper,
)

EXECUTION_PROPERTIES = {
    "execution_mode": {
        "enum": ["wait", "background"],
        "type": "string",
        "description": "Wait up to wait_seconds, or return immediately while the tool continues.",
        "default": "wait",
    },
    "wait_seconds": {
        "type": "number",
        "description": "Maximum initial wait in seconds; the tool continues after this time.",
        "default": 60.0,
    },
}


class Filters(TypedDict):
    category: str
    active: bool


def search(
    query: Annotated[str, "Text to search"],
    filters: Filters,
    limit: int = 10,
    mode: Literal["exact", "fuzzy"] = "exact",
    tags: list[str] | None = None,
) -> list[str]:
    """Search indexed documents."""
    return [query]


def test_map_tool_builds_openai_function_schema_from_annotations() -> None:
    mapped = OpenAIChatCompletionToolMapper.map_tool(Tool(search))

    assert mapped == {
        "type": "function",
        "function": {
            "name": "search",
            "description": "Search indexed documents.",
            "parameters": {
                "type": "object",
                "properties": {
                    **EXECUTION_PROPERTIES,
                    "query": {"type": "string", "description": "Text to search"},
                    "filters": {
                        "type": "object",
                        "properties": {
                            "category": {"type": "string"},
                            "active": {"type": "boolean"},
                        },
                        "additionalProperties": False,
                        "required": ["category", "active"],
                    },
                    "limit": {"type": "integer", "default": 10},
                    "mode": {
                        "enum": ["exact", "fuzzy"],
                        "type": "string",
                        "default": "exact",
                    },
                    "tags": {
                        "anyOf": [
                            {"type": "array", "items": {"type": "string"}},
                            {"type": "null"},
                        ],
                        "default": None,
                    },
                },
                "additionalProperties": False,
                "required": ["query", "filters"],
            },
        },
    }


def test_map_tools_preserves_registry_order() -> None:
    def count(value: int) -> int:
        return value

    mapped = OpenAIChatCompletionToolMapper.map_tools(
        Tools([Tool(search), Tool(count)])
    )

    assert [item["function"]["name"] for item in mapped] == ["search", "count"]


def test_map_tool_supports_bound_partial_parameters(tmp_path: Path) -> None:
    def run(work_root: Path, query: Annotated[str, "Query to run"]) -> str:
        return str(work_root / query)

    mapped = OpenAIChatCompletionToolMapper.map_tool(
        Tool(partial(run, tmp_path), name="run")
    )

    assert mapped["function"]["parameters"] == {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Query to run"},
            **EXECUTION_PROPERTIES,
        },
        "additionalProperties": False,
        "required": ["query"],
    }


def test_map_tool_omits_non_json_default_and_empty_required_list() -> None:
    opaque_default = object()

    def inspect_value(value: object = opaque_default) -> object:
        return value

    mapped = OpenAIChatCompletionToolMapper.map_tool(Tool(inspect_value))
    parameters = mapped["function"]["parameters"]

    assert parameters == {
        "type": "object",
        "properties": {"value": {}, **EXECUTION_PROPERTIES},
        "additionalProperties": False,
    }
    assert "description" not in mapped["function"]


@pytest.mark.parametrize(
    ("method", "value", "message"),
    [
        (OpenAIChatCompletionToolMapper.map_tools, object(), "Tools instance"),
        (OpenAIChatCompletionToolMapper.map_tool, object(), "Tool instance"),
    ],
)
def test_map_rejects_invalid_registry_types(
    method, value: object, message: str
) -> None:
    with pytest.raises(TypeError, match=message):
        method(value)


def test_map_tool_rejects_missing_annotations() -> None:
    def untyped(value):
        return value

    with pytest.raises(TypeError, match="must have a type annotation"):
        OpenAIChatCompletionToolMapper.map_tool(Tool(untyped))


def test_map_tool_rejects_variadic_parameters() -> None:
    def variadic(*values: str) -> None:
        pass

    with pytest.raises(TypeError, match="unsupported parameter"):
        OpenAIChatCompletionToolMapper.map_tool(Tool(variadic))


def test_resolve_tool_call_finds_tool_and_validates_arguments() -> None:
    registered = Tool(search)
    tool_call = ChatCompletionMessageFunctionToolCall(
        id="call_123",
        type="function",
        function=OpenAIFunction(
            name="search",
            arguments='{"query":"docs","filters":{"category":"api","active":true}}',
        ),
    )

    tool, arguments = OpenAIChatCompletionToolMapper.resolve_tool_call(
        tool_call,
        Tools([registered]),
    )

    assert tool is registered
    assert arguments == {
        "query": "docs",
        "filters": {"category": "api", "active": True},
    }


def test_resolve_tool_call_rejects_unknown_tool() -> None:
    tool_call = ChatCompletionMessageFunctionToolCall(
        id="call_123",
        type="function",
        function=OpenAIFunction(name="missing", arguments="{}"),
    )

    with pytest.raises(LookupError, match="not registered"):
        OpenAIChatCompletionToolMapper.resolve_tool_call(
            tool_call,
            Tools([Tool(search)]),
        )


def test_find_tool_supports_openai_tool_call() -> None:
    registered = Tool(search)
    tool_call = ChatCompletionMessageFunctionToolCall(
        id="call_123",
        type="function",
        function=OpenAIFunction(name="search", arguments="{}"),
    )

    assert (
        OpenAIChatCompletionToolMapper.find_tool(tool_call, Tools([registered]))
        is registered
    )


def test_find_tool_rejects_non_openai_function_call() -> None:
    tool_call = cast(ChatCompletionMessageFunctionToolCall, object())

    with pytest.raises(TypeError, match="OpenAI function tool call"):
        OpenAIChatCompletionToolMapper.find_tool(tool_call, Tools([Tool(search)]))


@pytest.mark.parametrize(
    ("arguments", "error_type", "message"),
    [
        ("not-json", ValueError, "not valid JSON"),
        ("[]", TypeError, "JSON object"),
        (
            '{"query":42,"filters":{"category":"api","active":true}}',
            TypeError,
            "must be a string",
        ),
        ('{"query":"docs"}', ValueError, "Invalid arguments"),
    ],
)
def test_resolve_tool_call_rejects_invalid_arguments(
    arguments: str,
    error_type: type[Exception],
    message: str,
) -> None:
    tool_call = ChatCompletionMessageFunctionToolCall(
        id="call_123",
        type="function",
        function=OpenAIFunction(name="search", arguments=arguments),
    )

    with pytest.raises(error_type, match=message):
        OpenAIChatCompletionToolMapper.resolve_tool_call(
            tool_call,
            Tools([Tool(search)]),
        )
