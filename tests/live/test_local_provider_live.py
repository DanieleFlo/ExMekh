from __future__ import annotations

import json
import os
from typing import Literal

import pytest
from dotenv import load_dotenv
from pydantic import BaseModel

from exmekh_lib.config.provider import DEFAULT_PROVIDER_SETTINGS
from exmekh_lib.core.models.assistant_message import ToolCall
from exmekh_lib.core.models.client import Client
from exmekh_lib.core.models.client_request import ClientRequest
from exmekh_lib.core.models.provider_response import ProviderResponse
from exmekh_lib.core.models.reasoning_effort import ReasoningEffortLMH
from exmekh_lib.core.models.system_message import SystemMessage
from exmekh_lib.core.models.tool import Tool
from exmekh_lib.core.models.tool_message import ToolMessage
from exmekh_lib.core.models.tools import Tools
from exmekh_lib.core.models.user_message import UserMessage
from tests.live.model_support import LiveModelProvider

pytestmark = pytest.mark.live


class RememberedFacts(BaseModel):
    project_code: Literal["ORBIT-17"]
    owner: Literal["Ada"]
    retry_limit: Literal[4]


class TicketSummary(BaseModel):
    ticket_id: Literal["EXMEKH-LIB-2048"]
    state: Literal["ready"]
    attempts: Literal[3]


def lookup_ticket(
    ticket_id: Literal["EXMEKH-LIB-1042", "EXMEKH-LIB-2048"],
) -> str:
    """Return the exact state and attempt count for a ticket."""
    tickets = {
        "EXMEKH-LIB-1042": {"state": "queued", "attempts": 2},
        "EXMEKH-LIB-2048": {"state": "ready", "attempts": 3},
    }
    return json.dumps({"ticket_id": ticket_id, **tickets[ticket_id]})


def build_live_client(system_prompt: str, tools: Tools | None = None) -> Client:
    load_dotenv()
    model = os.getenv("EXMEKH_LIB_LIVE_MODEL", DEFAULT_PROVIDER_SETTINGS.model)
    return Client(
        provider=LiveModelProvider(
            model=model, reasoning_effort=ReasoningEffortLMH.LOW
        ),
        system_message=SystemMessage(system_prompt),
        tools=tools,
    )


def require_ticket_calls(
    response: ProviderResponse,
    expected_ticket_id: str,
) -> list[ToolCall]:
    tool_calls = response.get_tool_calls() or []
    assert any(
        isinstance(call, ToolCall)
        and call.function.name == "lookup_ticket"
        and call.function.arguments.get("ticket_id") == expected_ticket_id
        for call in tool_calls
    ), f"MODEL: expected lookup_ticket({expected_ticket_id!r}) among {tool_calls!r}"
    assert all(
        isinstance(call, ToolCall) and call.function.name == "lookup_ticket"
        for call in tool_calls
    ), f"CLIENT: unexpected function calls: {tool_calls!r}"
    return tool_calls


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
    first_text = first_response.get_message()
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
    second_text = second_response.get_message()
    assert isinstance(second_text, str), "CLIENT: expected a text response"
    assert "orbit-17" in second_text.casefold() and "ada" in second_text.casefold(), (
        f"MODEL: chained facts were not recalled: {second_text!r}"
    )

    structured_response = client.call_structured(
        ClientRequest(
            UserMessage("Return all three stored facts in the requested JSON fields."),
            tool_choice="none",
        ),
        RememberedFacts,
    )
    assert structured_response.get_message() == RememberedFacts(
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
        "report only values returned by that tool. Keep replies to one short sentence.",
        tools=tools,
    )

    first_response = client.call(
        ClientRequest(
            UserMessage(
                "Look up ticket EXMEKH-LIB-1042 with the tool. Make exactly one call."
            ),
            tool_choice="required",
        )
    )
    first_calls = require_ticket_calls(first_response, "EXMEKH-LIB-1042")
    first_tool_messages = [
        ToolMessage(
            lookup_ticket(call.function.arguments["ticket_id"]),
            tool_call_id=call.id,
            name=call.get_name(),
        )
        for call in first_calls
    ]
    for message in first_tool_messages:
        client.add_tool_response(message)

    first_continuation = client.continue_after_tools(tool_choice="none")
    first_continuation_text = first_continuation.get_message()
    assert isinstance(first_continuation_text, str), (
        "CLIENT: expected text after the first tool response"
    )
    assert (
        "queued" in first_continuation_text.lower() and "2" in first_continuation_text
    ), f"MODEL: first tool result was not reflected: {first_continuation_text!r}"

    second_response = client.call(
        ClientRequest(
            UserMessage(
                "Now look up EXMEKH-LIB-2048. Again, make exactly one tool call."
            ),
            tool_choice="required",
        )
    )
    second_calls = require_ticket_calls(second_response, "EXMEKH-LIB-2048")
    second_tool_messages = [
        ToolMessage(
            lookup_ticket(call.function.arguments["ticket_id"]),
            tool_call_id=call.id,
            name=call.get_name(),
        )
        for call in second_calls
    ]
    for message in second_tool_messages:
        client.add_tool_response(message)

    structured_response = client.continue_after_tools_structured(
        TicketSummary,
        tool_choice="none",
    )
    assert structured_response.get_message() == TicketSummary(
        ticket_id="EXMEKH-LIB-2048",
        state="ready",
        attempts=3,
    ), "MODEL: structured result differs from the second tool response"

    history = client.get_history()
    expected_roles = ["system", "user", "assistant"]
    expected_roles.extend("tool" for _ in first_tool_messages)
    expected_roles.extend(["assistant", "user", "assistant"])
    expected_roles.extend("tool" for _ in second_tool_messages)
    expected_roles.append("assistant")
    assert [message.get_role_name() for message in history] == expected_roles, (
        "CLIENT: two-round tool history has an unexpected role order"
    )
    for call, message in zip(
        first_calls + second_calls,
        first_tool_messages + second_tool_messages,
        strict=True,
    ):
        assert message in history and message.get_tool_call_id() == call.id, (
            "CLIENT: tool response or its call id was lost"
        )
