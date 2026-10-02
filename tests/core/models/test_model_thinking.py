from dataclasses import FrozenInstanceError

import pytest

from exmekh_lib.core.models.model_thinking import ModelThinking


def test_init_uses_none_by_default() -> None:
    assert ModelThinking() == ModelThinking(None)


def test_init_preserves_thoughts() -> None:
    assert ModelThinking("reasoning").thoughts == "reasoning"


def test_model_thinking_is_immutable() -> None:
    thinking = ModelThinking()

    with pytest.raises(FrozenInstanceError):
        thinking.thoughts = "changed"


def test_init_rejects_non_string_thoughts() -> None:
    with pytest.raises(TypeError, match="string or None"):
        ModelThinking(42)
