import pytest

from qs.config.local_provider import LocalProviderConfig, ReasoningEffort


@pytest.mark.parametrize("effort", [None, ReasoningEffort.MEDIUM, ReasoningEffort.Max])
def test_config_preserves_explicit_non_default_values(effort) -> None:
    config = LocalProviderConfig(
        model="model-with-custom-revision",
        api_key_env_var="QS_TEST_KEY",
        base_url_env_var="QS_TEST_ENDPOINT",
        reasoning_effort=effort,
    )

    assert config.model == "model-with-custom-revision"
    assert config.api_key_env_var == "QS_TEST_KEY"
    assert config.base_url_env_var == "QS_TEST_ENDPOINT"
    assert config.reasoning_effort is effort


@pytest.mark.parametrize(
    ("overrides", "error_type"),
    [
        ({"model": ""}, ValueError),
        ({"model": "   "}, ValueError),
        ({"model": None}, ValueError),
        ({"api_key_env_var": ""}, ValueError),
        ({"base_url_env_var": 7}, ValueError),
        ({"reasoning_effort": "high"}, TypeError),
    ],
)
def test_config_rejects_malformed_values(overrides, error_type) -> None:
    values = {
        "model": "valid-model",
        "api_key_env_var": "VALID_KEY",
        "base_url_env_var": "VALID_URL",
        "reasoning_effort": ReasoningEffort.LOW,
    }
    values.update(overrides)

    with pytest.raises(error_type):
        LocalProviderConfig(**values)
