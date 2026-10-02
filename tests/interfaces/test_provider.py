from pathlib import Path

import pytest
from pydantic import BaseModel

from exmekh_lib.core.cache.cache_manager import CacheManager
from exmekh_lib.core.cache.session_cache import SessionCache
from exmekh_lib.core.models.assistant_message import AssistantMessage
from exmekh_lib.core.models.provider_request import ProviderRequest
from exmekh_lib.core.models.provider_response import ProviderResponse
from exmekh_lib.core.models.structured_assistant_message import (
    StructuredAssistantMessage,
)
from exmekh_lib.core.models.structured_provider_request import StructuredProviderRequest
from exmekh_lib.core.models.structured_provider_response import (
    StructuredProviderResponse,
)
from exmekh_lib.core.models.system_message import SystemMessage
from exmekh_lib.interfaces.provider import Provider
from exmekh_lib.interfaces.tokenizer import Tokenizer


class StubTokenizer(Tokenizer):
    def count_tokens(self, text: str) -> int:
        return len(text)

    def truncate_text(self, text: str, start_tokens: int, end_tokens: int = 0) -> str:
        return text[:start_tokens] + (text[-end_tokens:] if end_tokens else "")

    def initialize(self) -> None:
        self.initialized = True

    def generate_id(self) -> str:
        return "one_two_three"

    def get_session(self) -> SessionCache:
        return CacheManager.get_session(
            "stub-provider", "default", Path(".exmekh_lib/tokenizers")
        )


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
    provider = ConcreteProvider(StubTokenizer())
    request = ProviderRequest([SystemMessage("rules")])

    assert provider.call(request).get_message() == "response"


def test_provider_structured_contract_preserves_response_type() -> None:
    provider = ConcreteProvider(StubTokenizer())
    request = StructuredProviderRequest([SystemMessage("rules")], StructuredResponse)

    content = provider.call_structured(request).get_message()

    assert content == StructuredResponse(value=42)


def test_provider_uses_only_the_tokenizer_contract() -> None:
    tokenizer = StubTokenizer()
    provider = ConcreteProvider(tokenizer)

    provider.get_tokenizer().initialize()

    assert tokenizer.initialized
    assert provider.get_tokenizer() is tokenizer
    assert provider.generate_id() == "one_two_three"


@pytest.mark.parametrize("tokenizer", [None, object()])
def test_provider_requires_a_tokenizer_instance(tokenizer) -> None:
    with pytest.raises(TypeError, match="Tokenizer"):
        ConcreteProvider(tokenizer)
