"""Reusable mapping services must not share schemas or tool registries across calls."""

import pytest
from openai.types.chat.chat_completion_message_function_tool_call import (
    ChatCompletionMessageFunctionToolCall,
)
from openai.types.responses import ResponseFunctionToolCall

from exmekh_lib.core.models.tool import Tool
from exmekh_lib.core.models.tools import Tools
from exmekh_lib.core.providers.openai_chat_completion_tool_mapper import (
    OpenAIChatCompletionToolMapper,
)
from exmekh_lib.core.providers.openai_response_tool_mapper import (
    OpenAIResponseToolMapper,
)


def echo(value: str) -> str:
    return value


@pytest.fixture(params=[OpenAIChatCompletionToolMapper, OpenAIResponseToolMapper])
def mapper_type(request):
    return request.param


@pytest.mark.parametrize("use_instance", [False, True])
def test_repeated_mapping_preserves_class_and_instance_apis_without_shared_schemas(
    mapper_type, use_instance
):
    mapper = mapper_type() if use_instance else mapper_type
    tools = Tools([Tool(echo)])
    first = mapper.map_tools(tools)[0]
    definition = first.get("function", first)
    definition["parameters"]["properties"]["value"]["type"] = "changed"

    second = mapper.map_tools(tools)[0]
    definition = second.get("function", second)
    assert definition["parameters"]["properties"]["value"] == {"type": "string"}


@pytest.mark.parametrize("use_instance", [False, True])
def test_repeated_resolution_uses_the_current_registry_and_fresh_arguments(
    mapper_type, use_instance
):
    mapper = mapper_type() if use_instance else mapper_type
    if mapper_type is OpenAIChatCompletionToolMapper:
        call = ChatCompletionMessageFunctionToolCall(
            id="call",
            type="function",
            function={"name": "echo", "arguments": '{"value":"hello"}'},
        )
    else:
        call = ResponseFunctionToolCall(
            call_id="call",
            type="function_call",
            name="echo",
            arguments='{"value":"hello"}',
        )
    first_tool, second_tool = Tool(echo), Tool(echo)
    selected, arguments = mapper.resolve_tool_call(call, Tools([first_tool]))
    assert selected is first_tool
    arguments["value"] = "changed"

    selected, arguments = mapper.resolve_tool_call(call, Tools([second_tool]))
    assert selected is second_tool
    assert arguments == {"value": "hello"}
