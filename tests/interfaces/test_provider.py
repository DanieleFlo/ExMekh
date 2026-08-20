from typing import Any

import pytest

from qs.core.models.provider_response import ProviderResponse
from qs.interfaces.provider import Provider


class ConcreteProvider(Provider):
    def _build_client(self, *args, **kwargs) -> Any:
        return object()

    def _check_provider(self) -> None:
        return None

    def call(self, *args, **kwargs) -> ProviderResponse:
        return ProviderResponse("response")


def test_provider_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        Provider()


def test_complete_provider_implementation_satisfies_contract() -> None:
    provider = ConcreteProvider()

    assert provider._build_client() is not None
    assert provider._check_provider() is None
    assert provider.call().response == "response"
