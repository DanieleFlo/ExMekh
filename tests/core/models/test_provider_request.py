import pytest

from qs.core.models.assistant_message import AssistantMessage
from qs.core.models.provider_request import ProviderRequest
from qs.core.models.provider_stream_event import (
    ProviderStreamEvent,
    ProviderStreamEventType,
)
from qs.core.models.system_message import SystemMessage
from qs.core.models.tool import Tool
from qs.core.models.tools import Tools
from qs.core.models.user_message import UserMessage


def test_get_messages_returns_messages_in_order() -> None:
    messages = [
        SystemMessage("non-default-rules"),
        UserMessage("non-default-question"),
        AssistantMessage("non-default-answer"),
    ]
    request = ProviderRequest(messages)

    result = request.get_messages()

    assert result == messages


def test_get_messages_is_isolated_from_external_list_mutation() -> None:
    system_message = SystemMessage("stable-rules")
    source = [system_message]
    request = ProviderRequest(source)

    source.append(UserMessage("source mutation"))
    result = request.get_messages()
    result.append(UserMessage("attempted external mutation"))

    assert request.get_messages() == [system_message]


def test_request_preserves_provider_options() -> None:
    def search(query: str) -> str:
        return query

    tools = Tools([Tool(search)])
    extra = {"temperature": 0.2}
    request = ProviderRequest(
        [SystemMessage("rules")],
        extra=extra,
        tool_choice="auto",
        tools=tools,
    )

    extra["temperature"] = 1
    returned_extra = request.get_extra()
    assert returned_extra is not None
    returned_extra["temperature"] = 2

    assert request.get_extra() == {"temperature": 0.2}
    assert request.get_tool_choice() == "auto"
    assert request.get_tools() is tools


def test_request_preserves_streaming_and_executes_callback() -> None:
    events = []
    request = ProviderRequest(
        [SystemMessage("rules")],
        stream=True,
        on_event=events.append,
    )
    event = ProviderStreamEvent(ProviderStreamEventType.START)

    request.execute_on_stream_event(event)

    assert request.get_streaming() is True
    assert events == [event]


def test_execute_on_stream_event_without_callback_is_a_noop() -> None:
    request = ProviderRequest([SystemMessage("rules")])
    event = ProviderStreamEvent(ProviderStreamEventType.START)

    assert request.execute_on_stream_event(event) is None


@pytest.mark.parametrize("messages", [None, [], [object()]])
def test_request_rejects_invalid_messages(messages) -> None:
    with pytest.raises((TypeError, ValueError)):
        ProviderRequest(messages)


def test_request_rejects_invalid_options() -> None:
    messages = [SystemMessage("rules")]

    with pytest.raises(TypeError, match="dictionary"):
        ProviderRequest(messages, extra="invalid")

    with pytest.raises(TypeError, match="Tool choice"):
        ProviderRequest(messages, tool_choice=object())

    with pytest.raises(TypeError, match="Tools instance"):
        ProviderRequest(messages, tools=object())

    with pytest.raises(TypeError, match="boolean"):
        ProviderRequest(messages, stream=1)

    with pytest.raises(TypeError, match="callable"):
        ProviderRequest(messages, on_event=object())

    with pytest.raises(ValueError, match="Reasoning effort override"):
        ProviderRequest(messages, reasoning_effort_override="  ")
