from __future__ import annotations

import json
import os
from typing import Literal

import pytest
from dotenv import load_dotenv
from pydantic import BaseModel

from qs.core.models.assistant_message import FunctionToolCall
from qs.core.models.client import Client
from qs.core.models.client_request import ClientRequest
from qs.core.models.provider_response import ProviderResponse
from qs.core.models.reasoning_effort import ReasoningEffortLMH
from qs.core.models.system_message import SystemMessage
from qs.core.models.tool import Tool
from qs.core.models.tool_message import ToolMessage
from qs.core.models.tools import Tools
from qs.core.models.user_message import UserMessage
from qs.providers.local import LocalProvider

pytestmark = pytest.mark.live


class RememberedFacts(BaseModel):
    project_code: Literal["ORBIT-17"]
    owner: Literal["Ada"]
    retry_limit: Literal[4]


class TicketSummary(BaseModel):
    ticket_id: Literal["QS-2048"]
    state: Literal["ready"]
    attempts: Literal[3]


def lookup_ticket(
    ticket_id: Literal["QS-1042", "QS-2048"],
) -> str:
    """Return the exact state and attempt count for a ticket."""
    tickets = {
        "QS-1042": {"state": "queued", "attempts": 2},
        "QS-2048": {"state": "ready", "attempts": 3},
    }
    return json.dumps({"ticket_id": ticket_id, **tickets[ticket_id]})


def build_live_client(system_prompt: str, tools: Tools | None = None) -> Client:
    load_dotenv()
    model = os.getenv("QS_LIVE_MODEL", "qwen3.5-9b-mtp")
    return Client(
        provider=LocalProvider(model=model, reasoning_effort=ReasoningEffortLMH.LOW),
        system_message=SystemMessage(system_prompt),
        tools=tools,
    )


def require_single_function_call(
    response: ProviderResponse,
    expected_ticket_id: str,
) -> FunctionToolCall:
    tool_calls = response.response.get_tool_calls()
    assert tool_calls is not None, "MODEL: expected one tool call, received none"
    assert len(tool_calls) == 1, (
        f"MODEL: expected one tool call, received {len(tool_calls)}"
    )

    tool_call = tool_calls[0]
    assert isinstance(tool_call, FunctionToolCall), (
        f"CLIENT: expected a mapped function tool call, got {type(tool_call)!r}"
    )
    assert tool_call.function.name == "lookup_ticket", (
        f"MODEL: expected lookup_ticket, got {tool_call.function.name!r}"
    )

    arguments = json.loads(tool_call.function.arguments)
    assert arguments == {"ticket_id": expected_ticket_id}, (
        f"MODEL: unexpected tool arguments: {arguments!r}"
    )
    return tool_call


def test_live_chained_messages_preserve_multiple_facts() -> None:
    client = build_live_client(
        "Keep the user's facts across turns. Follow the requested output format."
    )

    first_response = client.call(
        ClientRequest(
            UserMessage(
                "Store these facts for later: project code ORBIT-17; owner Ada; "
                "retry limit 4. Reply only ACK."
            ),
            tool_choice="none",
        )
    )
    first_text = first_response.response.get_message()
    assert isinstance(first_text, str) and first_text.strip(), (
        "MODEL: the acknowledgement response was empty"
    )

    second_response = client.call(
        ClientRequest(
            UserMessage(
                "Using only the conversation, state the project code and owner "
                "in one short sentence."
            ),
            tool_choice="none",
        )
    )
    second_text = second_response.response.get_message()
    assert isinstance(second_text, str), "CLIENT: expected a text response"
    assert "ORBIT-17" in second_text and "Ada" in second_text, (
        f"MODEL: chained facts were not recalled: {second_text!r}"
    )

    structured_response = client.call_structured(
        ClientRequest(
            UserMessage("Return all three stored facts in the requested JSON fields."),
            tool_choice="none",
        ),
        RememberedFacts,
    )
    assert structured_response.response.get_message() == RememberedFacts(
        project_code="ORBIT-17",
        owner="Ada",
        retry_limit=4,
    ), "MODEL: structured facts differ from the conversation"

    assert [message.get_role_name() for message in client.get_history()] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
        "assistant",
    ], "CLIENT: chained call history has an unexpected role order"


def test_live_two_tool_round_trips_keep_history_and_ids() -> None:
    tools = Tools([Tool(lookup_ticket)])
    client = build_live_client(
        "Use lookup_ticket whenever a ticket is requested. After a tool result, "
        "report only values returned by that tool.",
        tools=tools,
    )

    first_response = client.call(
        ClientRequest(
            UserMessage("Look up ticket QS-1042 with the tool. Make exactly one call."),
            tool_choice="required",
        )
    )
    first_call = require_single_function_call(first_response, "QS-1042")
    first_tool_message = ToolMessage(
        lookup_ticket("QS-1042"),
        tool_call_id=first_call.id,
    )
    client.add_tool_response(first_tool_message)

    first_continuation = client.continue_after_tools(tool_choice="none")
    first_continuation_text = first_continuation.response.get_message()
    assert isinstance(first_continuation_text, str), (
        "CLIENT: expected text after the first tool response"
    )
    assert (
        "queued" in first_continuation_text.lower() and "2" in first_continuation_text
    ), f"MODEL: first tool result was not reflected: {first_continuation_text!r}"

    second_response = client.call(
        ClientRequest(
            UserMessage("Now look up QS-2048. Again, make exactly one tool call."),
            tool_choice="required",
        )
    )
    second_call = require_single_function_call(second_response, "QS-2048")
    second_tool_message = ToolMessage(
        lookup_ticket("QS-2048"),
        tool_call_id=second_call.id,
    )
    client.add_tool_response(second_tool_message)

    structured_response = client.continue_after_tools_structured(
        TicketSummary,
        tool_choice="none",
    )
    assert structured_response.response.get_message() == TicketSummary(
        ticket_id="QS-2048",
        state="ready",
        attempts=3,
    ), "MODEL: structured result differs from the second tool response"

    history = client.get_history()
    assert [message.get_role_name() for message in history] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
        "user",
        "assistant",
        "tool",
        "assistant",
    ], "CLIENT: two-round tool history has an unexpected role order"
    assert history[3] is first_tool_message and history[7] is second_tool_message, (
        "CLIENT: tool responses were not retained in their original positions"
    )
    assert first_tool_message.get_tool_call_id() == first_call.id, (
        "CLIENT: first tool response lost its call id"
    )
    assert second_tool_message.get_tool_call_id() == second_call.id, (
        "CLIENT: second tool response lost its call id"
    )
