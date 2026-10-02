from dataclasses import FrozenInstanceError

import pytest

from exmekh_lib.core.models.model_usage import ModelUsage


def test_init_uses_none_for_all_token_counts() -> None:
    usage = ModelUsage()

    assert usage == ModelUsage(None, None, None)


def test_init_preserves_token_counts() -> None:
    usage = ModelUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)

    assert usage.prompt_tokens == 10
    assert usage.completion_tokens == 5
    assert usage.total_tokens == 15


def test_model_usage_is_immutable() -> None:
    usage = ModelUsage()

    with pytest.raises(FrozenInstanceError):
        usage.total_tokens = 1


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("prompt_tokens", "prompt_tokens"),
        ("completion_tokens", "completion_tokens"),
        ("total_tokens", "total_tokens"),
    ],
)
def test_init_rejects_non_integer_token_counts(field: str, message: str) -> None:
    with pytest.raises(TypeError, match=message):
        ModelUsage(**{field: "invalid"})
