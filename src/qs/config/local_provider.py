from __future__ import annotations

import enum

from qs.interfaces.provider_config import ProviderConfig


class ReasoningEffort(enum.Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    XHIGH = "xhigh"
    Max = "max"


class LocalProviderConfig(ProviderConfig):
    def __init__(
        self,
        model: str,
        api_key_env_var: str = "API_KEY",
        base_url_env_var: str = "BASE_URL",
        reasoning_effort: ReasoningEffort | None = None,
    ) -> None:
        if not isinstance(model, str) or not model.strip():
            raise ValueError("Model must be a non-empty string")
        if not isinstance(api_key_env_var, str) or not api_key_env_var.strip():
            raise ValueError("API key environment variable must be a non-empty string")
        if not isinstance(base_url_env_var, str) or not base_url_env_var.strip():
            raise ValueError("Base URL environment variable must be a non-empty string")
        if reasoning_effort is not None and not isinstance(
            reasoning_effort, ReasoningEffort
        ):
            raise TypeError("Reasoning effort must be a ReasoningEffort or None")

        self.model = model
        self.api_key_env_var = api_key_env_var
        self.base_url_env_var = base_url_env_var
        self.reasoning_effort = reasoning_effort
