from __future__ import annotations

from qs.config.local_provider import LocalProviderConfig
from qs.interfaces.provider_config import ProviderConfig
from qs.providers.local import LocalProvider


class ProviderMapper:
    @staticmethod
    def provider(config_provider: ProviderConfig) -> LocalProvider:
        if isinstance(config_provider, LocalProviderConfig):
            return LocalProvider(config_provider)
        raise ValueError("Unsupported provider configuration")
