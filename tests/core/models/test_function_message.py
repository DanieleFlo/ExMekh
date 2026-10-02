from typing import cast

import pytest

from exmekh_lib.core.models.function_message import FunctionMessage


def test_init_uses_function_defaults() -> None:
    message = FunctionMessage("result", "calculate")

    assert message.get_role_name() == "function"
    assert message.get_name() == "calculate"


def test_get_message_returns_content() -> None:
    message = FunctionMessage("result", "calculate")

    assert message.get_message() == "result"


def test_get_name_returns_custom_name() -> None:
    message = FunctionMessage("result", "calculate")

    assert message.get_name() == "calculate"


def test_message_dump_returns_provider_fields() -> None:
    message = FunctionMessage("result", "calculate")

    assert message.message_dump() == {"role": "function", "content": "result"}


@pytest.mark.parametrize(
    ("content", "name", "message"),
    [
        (object(), "calculate", "content must be a string"),
        ("result", object(), "name must be a string"),
    ],
)
def test_init_rejects_invalid_fields(
    content: object, name: object, message: str
) -> None:
    with pytest.raises(TypeError, match=message):
        FunctionMessage(cast(str, content), cast(str, name))
