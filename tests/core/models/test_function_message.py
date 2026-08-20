from qs.core.models.function_message import FunctionMessage


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
