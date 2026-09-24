from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class ProviderSettings:
    """User-facing defaults for the local OpenAI-compatible provider."""

    model: str = "Qwen3.8-27B-UD-Q4_K_S"
    api: Literal["chat_completions", "responses"] = "responses"
    local_provider_default_api: Literal["chat_completions", "responses"] = (
        "chat_completions"
    )
    api_key_env_var: str = "API_KEY"
    base_url_env_var: str = "BASE_URL"
    fallback_api_key: str = "local-api-key"

    @classmethod
    def from_environment(cls) -> ProviderSettings:
        defaults = cls()
        return cls(
            model=os.getenv("QS_MODEL", defaults.model),
            api=os.getenv("QS_API", defaults.api),
            api_key_env_var=defaults.api_key_env_var,
            base_url_env_var=defaults.base_url_env_var,
            fallback_api_key=defaults.fallback_api_key,
        )


DEFAULT_PROVIDER_SETTINGS = ProviderSettings()
