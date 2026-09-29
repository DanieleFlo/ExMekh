from dataclasses import FrozenInstanceError
from types import MappingProxyType

import pytest

from qs.core.models.client_request import ClientRequest
from qs.core.models.user_message import UserMessage
from tests.event_handler import as_provider_handler


def test_request_preserves_message_and_options() -> None:
    message = UserMessage("question")
    extra = {"temperature": 0.2}

    request = ClientRequest(
        message,
        extra=extra,
        tool_choice="auto",
    )
    extra["temperature"] = 1

    assert request.message is message
    assert request.extra == {"temperature": 0.2}
    assert request.tool_choice == "auto"


def test_request_preserves_streaming_callback() -> None:
    def callback(event) -> None:
        del event

    handler = as_provider_handler(callback)
    request = ClientRequest(
        UserMessage("question"),
        streaming=True,
        on_event=handler,
    )

    assert request.streaming is True
    assert request.on_event is handler


def test_request_copies_extra_from_a_read_only_mapping() -> None:
    source = {"temperature": 0.2}
    request = ClientRequest(UserMessage("question"), extra=MappingProxyType(source))
    source["temperature"] = 1

    assert request.extra == {"temperature": 0.2}


def test_request_is_frozen() -> None:
    request = ClientRequest(UserMessage("question"))

    with pytest.raises(FrozenInstanceError):
        request.message = UserMessage("replacement")


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"message": object()}, "Message instance"),
        ({"message": UserMessage("question"), "extra": object()}, "mapping"),
        ({"message": UserMessage("question"), "tool_choice": object()}, "string"),
        ({"message": UserMessage("question"), "streaming": 1}, "boolean"),
        (
            {"message": UserMessage("question"), "on_event": object()},
            "ProviderEventHandler",
        ),
    ],
)
def test_request_rejects_invalid_values(arguments, message: str) -> None:
    with pytest.raises(TypeError, match=message):
        ClientRequest(**arguments)
