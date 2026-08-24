from typing import cast

import pytest

from qs.core.models.system_message import SystemMessage


def test_init_uses_system_role_and_empty_name_by_default() -> None:
    message = SystemMessage("instructions")

    assert message.get_role_name() == "system"
    assert message.get_name() == ""


def test_get_message_returns_content() -> None:
    message = SystemMessage("instructions")

    assert message.get_message() == "instructions"


def test_get_name_returns_custom_name() -> None:
    message = SystemMessage("instructions", name="policy")

    assert message.get_name() == "policy"


def test_message_dump_returns_provider_fields() -> None:
    message = SystemMessage("instructions", name="policy")

    assert message.message_dump() == {
        "role": "system",
        "content": "instructions",
    }


@pytest.mark.parametrize(
    ("content", "name", "message"),
    [
        (object(), "", "content must be a string"),
        ("instructions", object(), "name must be a string"),
    ],
)
def test_init_rejects_invalid_fields(
    content: object, name: object, message: str
) -> None:
    with pytest.raises(TypeError, match=message):
        SystemMessage(cast(str, content), name=cast(str, name))
