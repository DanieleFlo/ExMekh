import pytest

from qs.core.models.system_message import SystemMessage
from qs.interfaces.message import Message


class ConcreteMessage(Message[str]):
    def get_message(self) -> str:
        return "content"

    def get_role_name(self) -> str:
        return "test"

    def message_dump(self) -> dict:
        return {"role": "test", "content": "content"}


def test_message_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        Message()


def test_complete_message_implementation_satisfies_contract() -> None:
    message = ConcreteMessage()

    assert message.get_message() == "content"
    assert message.get_role_name() == "test"
    assert message.message_dump() == {"role": "test", "content": "content"}


def test_copy_preserves_concrete_message_without_sharing_state() -> None:
    message = SystemMessage("rules", name="system")

    copied_message = message.copy()

    assert isinstance(copied_message, SystemMessage)
    assert copied_message is not message
    assert copied_message.message_dump() == message.message_dump()
