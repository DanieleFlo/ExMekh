from dataclasses import FrozenInstanceError

import pytest

from qs.core.models.model_usage import ModelUsage


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
