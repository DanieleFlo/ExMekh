from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel

from qs.core.models.provider_request import ProviderRequest
from qs.core.models.provider_response import ProviderResponse
from qs.core.models.structured_provider_request import StructuredProviderRequest
from qs.core.models.structured_provider_response import StructuredProviderResponse


class Provider(ABC):
    @abstractmethod
    def call(self, request: ProviderRequest) -> ProviderResponse:
        pass

    @abstractmethod
    def call_structured[ResponseModel: BaseModel](
        self,
        request: StructuredProviderRequest[ResponseModel],
    ) -> StructuredProviderResponse[ResponseModel]:
        pass
