from __future__ import annotations

from qs.interfaces.provider_config import ProviderConfig


class LocalProviderConfig(ProviderConfig):
    def __init__(self, model: str, api_key_env_var: str = "API_KEY", base_url_env_var: str = "BASE_URL") -> None:
        self.model = model
        self.api_key_env_var = api_key_env_var
        self.base_url_env_var = base_url_env_var