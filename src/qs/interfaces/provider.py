from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from qs.core.models.provider_request import ProviderRequest
from qs.core.models.provider_response import ProviderResponse


class Provider(ABC):
    @abstractmethod
    def _build_client(self, *args, **kwargs) -> Any:
        pass

    @abstractmethod
    def _check_provider(self) -> None:
        pass

    @abstractmethod
    def call(self, request: ProviderRequest) -> ProviderResponse:
        pass
