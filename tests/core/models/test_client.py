from unittest.mock import MagicMock, patch

import pytest

from qs.config.local_provider import LocalProviderConfig
from qs.core.models.client import Client
from qs.core.providers.provider_mapper import ProviderMapper
from qs.interfaces.provider import Provider


def test_init_accepts_provider_returned_by_mapper() -> None:
    config = LocalProviderConfig(model="local-model")
    provider = MagicMock(spec=Provider)

    with patch.object(ProviderMapper, "provider", return_value=provider) as mapper:
        client = Client(config_provider=config)

    assert client.config_provider is config
    assert client._provider is provider
    mapper.assert_called_once_with(config_provider=config)


def test_init_rejects_non_provider_returned_by_mapper() -> None:
    config = LocalProviderConfig(model="local-model")

    with (
        patch.object(ProviderMapper, "provider", return_value=object()),
        pytest.raises(TypeError, match="Expected a Provider instance"),
    ):
        Client(config_provider=config)