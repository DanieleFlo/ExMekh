from unittest.mock import patch

import pytest

from qs.config.local_provider import LocalProviderConfig
from qs.core.providers.provider_mapper import ProviderMapper
from qs.interfaces.provider_config import ProviderConfig


def test_provider_builds_local_provider_from_explicit_config() -> None:
    config = LocalProviderConfig(
        model="mapped-model",
        api_key_env_var="MAPPED_KEY",
        base_url_env_var="MAPPED_URL",
    )
    expected = object()

    with patch(
        "qs.core.providers.provider_mapper.LocalProvider", return_value=expected
    ) as local_provider:
        result = ProviderMapper.provider(config)

    assert result is expected
    local_provider.assert_called_once_with(config)


def test_provider_rejects_unknown_config_without_evaluating_repr() -> None:
    class HostileConfig(ProviderConfig):
        def __repr__(self) -> str:
            raise RuntimeError("repr must not execute")

    with pytest.raises(ValueError, match="Unsupported provider configuration"):
        ProviderMapper.provider(HostileConfig())
