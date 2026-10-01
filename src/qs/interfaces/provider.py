from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pydantic import BaseModel

from qs.core.models.provider_request import ProviderRequest
from qs.core.models.provider_response import ProviderResponse
from qs.core.models.structured_provider_request import StructuredProviderRequest
from qs.core.models.structured_provider_response import StructuredProviderResponse
from qs.core.models.tools import Tools
from qs.interfaces.tokenizer import Tokenizer

if TYPE_CHECKING:
    from qs.core.cache.session_cache import SessionCache


class Provider(ABC):
    def __init__(self, tokenizer: Tokenizer) -> None:
        if not isinstance(tokenizer, Tokenizer):
            raise TypeError("Tokenizer must be a Tokenizer instance")
        self._tokenizer = tokenizer

    def get_tokenizer(self) -> Tokenizer:
        return self._tokenizer

    def generate_id(self) -> str:
        return self._tokenizer.generate_id()

    def get_session(self) -> SessionCache:
        """Access the runtime session cache of the underlying tokenizer."""
        return self._tokenizer.get_session()

    def get_tools_text(self, tools: Tools | None) -> str:
        """Return the serialized tools request field, or empty text if omitted."""
        raise NotImplementedError(
            "This provider does not implement tools request serialization"
        )

    @abstractmethod
    def call(self, request: ProviderRequest) -> ProviderResponse:
        pass  # pragma: no cover

    @abstractmethod
    def call_structured[ResponseModel: BaseModel](
        self,
        request: StructuredProviderRequest[ResponseModel],
    ) -> StructuredProviderResponse[ResponseModel]:
        pass  # pragma: no cover
