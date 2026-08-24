import pytest

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
