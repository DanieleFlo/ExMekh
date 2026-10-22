from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from pydantic import BaseModel

from exmekh_lib.config.interface_text import INTERFACE_TEXT
from exmekh_lib.config.native_suite import NATIVE_SUITE
from exmekh_lib.core.models.provider_request import ProviderRequest
from exmekh_lib.core.models.provider_response import ProviderResponse
from exmekh_lib.core.models.structured_provider_request import StructuredProviderRequest
from exmekh_lib.core.models.structured_provider_response import (
    StructuredProviderResponse,
)
from exmekh_lib.core.models.tools import Tools
from exmekh_lib.interfaces.tokenizer import Tokenizer

if TYPE_CHECKING:
    from exmekh_lib.core.cache.session_cache import SessionCache
    from exmekh_lib.core.models.native_suite import NativeSuiteClient


class ProviderRequestRejectedError(RuntimeError):
    """The provider rejected the input before producing a response."""


class ProviderResponseIncompleteError(RuntimeError):
    """Generation ended without a complete response, possibly at its output limit."""


class Provider(ABC):
    def __init__(self, tokenizer: Tokenizer) -> None:
        if not isinstance(tokenizer, Tokenizer):
            raise TypeError(INTERFACE_TEXT.tokenizer_must_be_a_tokenizer_instance)
        self._tokenizer = tokenizer

    def get_tokenizer(self) -> Tokenizer:
        return self._tokenizer

    def native_suite(self) -> NativeSuiteClient:
        raise NotImplementedError(NATIVE_SUITE.unsupported_provider)

    def generate_id(self) -> str:
        return self._tokenizer.generate_id()

    def get_session(self) -> SessionCache:
        """Access the runtime session cache of the underlying tokenizer."""
        return self._tokenizer.get_session()

    def get_tools_text(self, tools: Tools | None) -> str:
        """Return the serialized tools request field, or empty text if omitted."""
        raise NotImplementedError(
            INTERFACE_TEXT.this_provider_does_not_implement_tools_request_serialization
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
