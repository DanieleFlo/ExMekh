from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel

from qs.core.models.provider_request import ProviderRequest
from qs.core.models.provider_response import ProviderResponse
from qs.core.models.structured_provider_request import StructuredProviderRequest
from qs.core.models.structured_provider_response import StructuredProviderResponse
from qs.interfaces.tokenizer import Tokenizer


class Provider(ABC):
    def __init__(self, tokenizer: Tokenizer) -> None:
        if not isinstance(tokenizer, Tokenizer):
            raise TypeError("Tokenizer must be a Tokenizer instance")
        self._tokenizer = tokenizer

    def get_tokenizer(self) -> Tokenizer:
        return self._tokenizer

    def generate_id(self) -> str:
        return self._tokenizer.generate_id()

    @abstractmethod
    def call(self, request: ProviderRequest) -> ProviderResponse:
        pass  # pragma: no cover

    @abstractmethod
    def call_structured[ResponseModel: BaseModel](
        self,
        request: StructuredProviderRequest[ResponseModel],
    ) -> StructuredProviderResponse[ResponseModel]:
        pass  # pragma: no cover
