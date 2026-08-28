import pytest
from pydantic import BaseModel

from qs.core.models.assistant_message import AssistantMessage
from qs.core.models.provider_request import ProviderRequest
from qs.core.models.provider_response import ProviderResponse
from qs.core.models.structured_assistant_message import StructuredAssistantMessage
from qs.core.models.structured_provider_request import StructuredProviderRequest
from qs.core.models.structured_provider_response import StructuredProviderResponse
from qs.core.models.system_message import SystemMessage
from qs.interfaces.provider import Provider


class StructuredResponse(BaseModel):
    value: int


class ConcreteProvider(Provider):
    def call(self, request: ProviderRequest) -> ProviderResponse:
        return ProviderResponse(AssistantMessage("response"))

    def call_structured[ResponseModel: BaseModel](
        self,
        request: StructuredProviderRequest[ResponseModel],
    ) -> StructuredProviderResponse[ResponseModel]:
        response_model = request.get_response_model()
        content = response_model.model_validate(
            {"value": 42}, strict=request.get_strict()
        )
        return StructuredProviderResponse(
            StructuredAssistantMessage(content),
        )


def test_provider_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        Provider()


def test_complete_provider_implementation_satisfies_contract() -> None:
    provider = ConcreteProvider()
    request = ProviderRequest([SystemMessage("rules")])

    assert provider.call(request).get_message() == "response"


def test_provider_structured_contract_preserves_response_type() -> None:
    provider = ConcreteProvider()
    request = StructuredProviderRequest([SystemMessage("rules")], StructuredResponse)

    content = provider.call_structured(request).get_message()

    assert content == StructuredResponse(value=42)
