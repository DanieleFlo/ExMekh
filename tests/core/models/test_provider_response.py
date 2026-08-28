import pytest

from qs.core.models.assistant_message import AssistantMessage
from qs.core.models.model_thinking import ModelThinking
from qs.core.models.model_usage import ModelUsage
from qs.core.models.provider_response import ProviderResponse


def test_init_uses_optional_metadata_defaults() -> None:
    message = AssistantMessage("answer")
    response = ProviderResponse(message)

    assert response.get_message() == "answer"
    assert response.get_role_name() == message.get_role_name()
    assert response.get_name() == message.get_name()
    assert response.get_tool_calls() is message.get_tool_calls()
    assert response.thinking is None
    assert response.usage is None


def test_init_preserves_response_metadata() -> None:
    thinking = ModelThinking("reasoning")
    usage = ModelUsage(10, 5, 15)
    message = AssistantMessage("answer")

    response = ProviderResponse(message, thinking=thinking, usage=usage)

    assert isinstance(response, AssistantMessage)
    assert response.get_message() == message.get_message()
    assert response.thinking is thinking
    assert response.usage is usage


def test_init_rejects_non_message_response() -> None:
    with pytest.raises(TypeError, match="Response must be an AssistantMessage"):
        ProviderResponse(object())


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ({"usage": object()}, "ModelUsage"),
        ({"thinking": object()}, "ModelThinking"),
    ],
)
def test_init_rejects_invalid_metadata(metadata: dict, message: str) -> None:
    with pytest.raises(TypeError, match=message):
        ProviderResponse(AssistantMessage("answer"), **metadata)


def test_init_rejects_response_without_text_content() -> None:
    with pytest.raises(TypeError, match="content must be a string"):
        ProviderResponse(AssistantMessage(None))
