from functools import partial
from pathlib import Path

import pytest
from openai.types.responses import (
    Response,
    ResponseFunctionToolCall,
    ResponseOutputMessage,
    ResponseOutputRefusal,
    ResponseOutputText,
    ResponseReasoningItem,
    ResponseUsage,
)
from openai.types.responses.response_reasoning_item import Content, Summary

from exmekh_lib.core.models.tool import Tool
from exmekh_lib.core.models.tools import Tools
from exmekh_lib.core.providers.from_openai_response_mapper import (
    FromOpenAIResponseMapper,
)


def make_response(*output, usage=None) -> Response:
    return Response.model_construct(output=list(output), usage=usage)


def test_map_assistant_message_maps_text_reasoning_usage_and_tool_calls(
    tmp_path: Path,
) -> None:
    def run(work_root: Path, count: int) -> str:
        return str(work_root / str(count))

    response = make_response(
        ResponseReasoningItem(
            id="reasoning_1",
            summary=[Summary(text="Think first", type="summary_text")],
            type="reasoning",
            content=None,
            encrypted_content=None,
            status="completed",
        ),
        ResponseOutputMessage(
            id="message_1",
            content=[
                ResponseOutputText(
                    annotations=[],
                    text="Calling tool",
                    type="output_text",
                    logprobs=[],
                )
            ],
            role="assistant",
            status="completed",
            type="message",
        ),
        ResponseFunctionToolCall(
            arguments='{"count":2}',
            call_id="call_123",
            name="run",
            type="function_call",
            id="item_1",
            status="completed",
        ),
        usage=ResponseUsage.model_construct(
            input_tokens=3,
            output_tokens=4,
            total_tokens=7,
        ),
    )
    tools = Tools([Tool(partial(run, tmp_path), name="run")])

    message = FromOpenAIResponseMapper.map_assistant_message(response, tools)

    assert message.get_message() == "Calling tool"
    assert message.get_tool_calls()[0].id == "call_123"
    assert message.get_tool_calls()[0].function.arguments == {"count": 2}
    assert FromOpenAIResponseMapper.extract_thinking(response).thoughts == "Think first"
    assert FromOpenAIResponseMapper.extract_usage(response).total_tokens == 7


def test_map_assistant_message_requires_tools_for_function_calls() -> None:
    response = make_response(
        ResponseFunctionToolCall(
            arguments="{}",
            call_id="call_123",
            name="run",
            type="function_call",
        )
    )

    with pytest.raises(ValueError, match="Tools are required"):
        FromOpenAIResponseMapper.map_assistant_message(response)


def test_extract_empty_metadata_and_reject_invalid_response() -> None:
    response = make_response()

    assert FromOpenAIResponseMapper.extract_usage(response).total_tokens is None
    assert FromOpenAIResponseMapper.extract_thinking(response) is None
    assert (
        FromOpenAIResponseMapper.map_assistant_message(response).get_message() is None
    )

    with pytest.raises(TypeError, match="OpenAI Response"):
        FromOpenAIResponseMapper.map_assistant_message(object())


def test_map_assistant_message_maps_refusal_content() -> None:
    response = make_response(
        ResponseOutputMessage(
            id="message_1",
            content=[ResponseOutputRefusal(refusal="Cannot answer", type="refusal")],
            role="assistant",
            status="completed",
            type="message",
        )
    )

    message = FromOpenAIResponseMapper.map_assistant_message(response)

    assert message.get_message() == "Cannot answer"


def test_extract_thinking_prefers_reasoning_content() -> None:
    response = make_response(
        ResponseReasoningItem(
            id="reasoning_1",
            summary=[Summary(text="Summary", type="summary_text")],
            type="reasoning",
            content=[
                Content(text="Full reasoning", type="reasoning_text"),
                Content(text="", type="reasoning_text"),
            ],
            encrypted_content=None,
            status="completed",
        )
    )

    thinking = FromOpenAIResponseMapper.extract_thinking(response)

    assert thinking.thoughts == "Full reasoning"


def test_map_message_content_ignores_unknown_part() -> None:
    message = ResponseOutputMessage.model_construct(content=[object()])

    assert FromOpenAIResponseMapper._map_message_content(message) == []
