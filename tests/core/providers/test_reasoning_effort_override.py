from qs.core.models.provider_request import ProviderRequest
from qs.core.models.system_message import SystemMessage
from qs.core.providers.to_openai_chat_completion_request_mapper import (
    ToOpenAIChatCompletionRequestMapper,
)
from qs.core.providers.to_openai_response_request_mapper import (
    ToOpenAIResponseRequestMapper,
)


def test_reasoning_override_applies_to_one_chat_request() -> None:
    request = ProviderRequest(
        [SystemMessage("Summarize")], reasoning_effort_override="none"
    )
    payload = ToOpenAIChatCompletionRequestMapper.map_request(
        request, model="model", reasoning_effort="high"
    )
    assert payload["reasoning_effort"] == "none"
    assert request.get_reasoning_effort_override() == "none"


def test_reasoning_override_applies_to_one_responses_request() -> None:
    request = ProviderRequest(
        [SystemMessage("Summarize")], reasoning_effort_override="none"
    )
    payload = ToOpenAIResponseRequestMapper.map_request(
        request, model="model", reasoning_effort="high"
    )
    assert payload["reasoning"] == {"effort": "none"}
    assert "tools" not in payload
