import pytest

from exmekh_lib.core.models.assistant_message import AssistantMessage
from exmekh_lib.core.models.model_thinking import ModelThinking
from exmekh_lib.core.models.model_usage import ModelUsage
from exmekh_lib.core.models.provider_response import ProviderResponse


def test_init_uses_optional_metadata_defaults() -> None:
    message = AssistantMessage("answer")
    response = ProviderResponse(message)

    assert response.get_message() == "answer"
    assert response.get_role_name() == message.get_role_name()
    assert response.get_name() == message.get_name()
    assert response.get_tool_calls() is message.get_tool_calls()
    assert response.thinking is None
    assert response.usage is None
    assert response.superseded_response is None


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
        ({"superseded_response": object()}, "AssistantMessage"),
    ],
)
def test_init_rejects_invalid_metadata(metadata: dict, message: str) -> None:
    with pytest.raises(TypeError, match=message):
        ProviderResponse(AssistantMessage("answer"), **metadata)


def test_init_preserves_response_without_content() -> None:
    response = ProviderResponse(AssistantMessage(None))

    assert response.get_message() is None


def test_superseded_response_is_evidence_without_executable_calls() -> None:
    completed = AssistantMessage("superseded")
    response = ProviderResponse(AssistantMessage(None), superseded_response=completed)
    assert response.superseded_response is completed
    assert response.get_message() is None
    assert not response.get_tool_calls()
