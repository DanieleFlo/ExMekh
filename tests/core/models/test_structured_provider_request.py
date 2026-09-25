from typing import cast

import pytest
from pydantic import BaseModel

from qs.core.models.structured_provider_request import StructuredProviderRequest
from qs.core.models.system_message import SystemMessage
from tests.event_handler import as_provider_handler


class Answer(BaseModel):
    value: int


def test_init_preserves_response_model_and_strict_default() -> None:
    request = StructuredProviderRequest([SystemMessage("rules")], Answer)

    assert request.get_response_model() is Answer
    assert request.get_strict() is True


def test_init_preserves_non_strict_mode_and_provider_options() -> None:
    request = StructuredProviderRequest(
        [SystemMessage("rules")],
        Answer,
        strict=False,
        extra={"temperature": 0.2},
        tool_choice="auto",
    )

    assert request.get_strict() is False
    assert request.get_extra() == {"temperature": 0.2}
    assert request.get_tool_choice() == "auto"


def test_init_preserves_streaming_options() -> None:
    def callback(event) -> None:
        del event

    handler = as_provider_handler(callback)
    request = StructuredProviderRequest(
        [SystemMessage("rules")],
        Answer,
        stream=True,
        on_event=handler,
    )

    assert request.get_streaming() is True
    assert request.on_event is handler


@pytest.mark.parametrize("response_model", [object, Answer(value=42)])
def test_init_rejects_non_pydantic_model_type(response_model) -> None:
    with pytest.raises(TypeError, match="Pydantic BaseModel type"):
        StructuredProviderRequest(
            [SystemMessage("rules")],
            cast(type[Answer], response_model),
        )


def test_init_rejects_non_boolean_strict_value() -> None:
    with pytest.raises(TypeError, match="Strict must be a boolean"):
        StructuredProviderRequest(
            [SystemMessage("rules")],
            Answer,
            strict=cast(bool, 1),
        )
