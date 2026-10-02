from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from exmekh_lib.core.models.assistant_message import (
    AssistantMessage,
    FunctionCall,
    ToolCall,
)
from exmekh_lib.core.models.client import Client
from exmekh_lib.core.models.client_request import ClientRequest
from exmekh_lib.core.models.provider_response import ProviderResponse
from exmekh_lib.core.models.system_message import SystemMessage
from exmekh_lib.core.models.tool_message import ToolMessage
from exmekh_lib.core.models.user_message import UserMessage
from exmekh_lib.interfaces.provider import Provider


class StructuredAnswer(BaseModel):
    value: int


def test_default_client_guard_rejects_steering_after_skill_return() -> None:
    provider = MagicMock(spec=Provider)
    client = Client(provider, SystemMessage("rules"))
    client.add_user_message(UserMessage("start"))
    client.add_tool_calls([ToolCall("skill-id", FunctionCall("skill", {}))])
    client.add_tool_response(ToolMessage("done", "skill-id", "skill"))

    with pytest.raises(ValueError, match="Agentic return must end"):
        client.continue_with_user_messages(
            [UserMessage("steering already delivered inside the skill")],
            extra={"agentic": {"operation": "return", "skill_call_id": "skill-id"}},
        )

    provider.call.assert_not_called()


def test_default_client_guard_accepts_skill_result_at_end() -> None:
    provider = MagicMock(spec=Provider)
    provider.call.return_value = ProviderResponse(AssistantMessage("done"))
    client = Client(provider, SystemMessage("rules"))
    client.add_user_message(UserMessage("start"))
    client.add_tool_calls([ToolCall("skill-id", FunctionCall("skill", {}))])
    client.add_tool_response(ToolMessage("done", "skill-id", "skill"))

    client.continue_after_tools(
        extra={"agentic": {"operation": "return", "skill_call_id": "skill-id"}}
    )

    provider.call.assert_called_once()


def test_default_client_guard_rejects_a_sibling_result_after_target_skill() -> None:
    provider = MagicMock(spec=Provider)
    client = Client(provider, SystemMessage("rules"))
    client.add_user_message(UserMessage("start"))
    client.add_tool_calls([ToolCall("first", FunctionCall("skill", {}))])
    client.add_tool_response(ToolMessage("first done", "first", "skill"))
    client.add_tool_calls([ToolCall("second", FunctionCall("skill", {}))])
    client.add_tool_response(ToolMessage("second done", "second", "skill"))

    with pytest.raises(ValueError, match="tool result for first"):
        client.continue_after_tools(
            extra={"agentic": {"operation": "return", "skill_call_id": "first"}}
        )

    provider.call.assert_not_called()


def test_default_client_guard_rejects_orphan_tool_result() -> None:
    provider = MagicMock(spec=Provider)
    client = Client(provider, SystemMessage("rules"))
    client.add_user_message(UserMessage("start"))
    client.add_tool_response(ToolMessage("orphan", "missing", "tool"))

    with pytest.raises(ValueError, match="Orphan or duplicate tool result"):
        client.continue_after_tools()

    provider.call.assert_not_called()


def test_default_client_guard_checks_structured_requests() -> None:
    provider = MagicMock(spec=Provider)
    client = Client(provider, SystemMessage("rules"))
    client.add_user_message(UserMessage("start"))
    client.add_tool_response(ToolMessage("orphan", "missing", "tool"))

    with pytest.raises(ValueError, match="Orphan or duplicate tool result"):
        client.call_structured(ClientRequest(UserMessage("continue")), StructuredAnswer)

    provider.call_structured.assert_not_called()
