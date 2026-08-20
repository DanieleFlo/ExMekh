from dataclasses import FrozenInstanceError

import pytest

from qs.core.models.model_thinking import ModelThinking


def test_init_uses_none_by_default() -> None:
    assert ModelThinking() == ModelThinking(None)


def test_init_preserves_thoughts() -> None:
    assert ModelThinking("reasoning").thoughts == "reasoning"


def test_model_thinking_is_immutable() -> None:
    thinking = ModelThinking()

    with pytest.raises(FrozenInstanceError):
        thinking.thoughts = "changed"
