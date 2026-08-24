import pytest
from pydantic import BaseModel

from qs.core.models.model_thinking import ModelThinking
from qs.core.models.model_usage import ModelUsage
from qs.core.models.structured_assistant_message import StructuredAssistantMessage
from qs.core.models.structured_provider_response import StructuredProviderResponse


class Answer(BaseModel):
    value: int


def test_init_preserves_response_and_metadata() -> None:
    message = StructuredAssistantMessage(Answer(value=42), name="agent", tool_calls=[])
    thinking = ModelThinking("reasoning")
    usage = ModelUsage(10, 5, 15)

    response = StructuredProviderResponse(
        message,
        thinking=thinking,
        usage=usage,
    )

    assert response.response is message
    assert response.thinking is thinking
    assert response.usage is usage


def test_init_uses_optional_metadata_defaults() -> None:
    response = StructuredProviderResponse(StructuredAssistantMessage(Answer(value=42)))

    assert response.thinking is None
    assert response.usage is None


def test_init_rejects_non_structured_message() -> None:
    with pytest.raises(TypeError, match="StructuredAssistantMessage"):
        StructuredProviderResponse(object())


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ({"usage": object()}, "ModelUsage"),
        ({"thinking": object()}, "ModelThinking"),
    ],
)
def test_init_rejects_invalid_metadata(metadata: dict, message: str) -> None:
    response = StructuredAssistantMessage(Answer(value=42))

    with pytest.raises(TypeError, match=message):
        StructuredProviderResponse(response, **metadata)


def test_init_rejects_non_model_response_content(monkeypatch) -> None:
    message = StructuredAssistantMessage(Answer(value=42))
    monkeypatch.setattr(message, "get_message", lambda: object())

    with pytest.raises(TypeError, match="Pydantic BaseModel"):
        StructuredProviderResponse(message)
