from qs.core.models.assistant_message import AssistantMessage
from qs.core.models.model_thinking import ModelThinking
from qs.core.models.model_usage import ModelUsage
from qs.core.models.provider_response import ProviderResponse


def test_init_uses_optional_metadata_defaults() -> None:
    message = AssistantMessage("answer")
    response = ProviderResponse(message)

    assert response.response is message
    assert response.thinking is None
    assert response.usage is None


def test_init_preserves_response_metadata() -> None:
    thinking = ModelThinking("reasoning")
    usage = ModelUsage(10, 5, 15)
    message = AssistantMessage("answer")

    response = ProviderResponse(message, thinking=thinking, usage=usage)

    assert response.response is message
    assert response.thinking is thinking
    assert response.usage is usage
