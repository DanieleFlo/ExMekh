from types import SimpleNamespace
from unittest.mock import patch

from qs.core.models.assistant_message import AssistantMessage
from qs.core.models.system_message import SystemMessage
from qs.core.models.user_message import UserMessage
from qs.main import main


def test_main_sends_default_messages_and_prints_response(capsys) -> None:
    with patch("qs.main.Client") as client_class:
        client = client_class.return_value
        client.call.return_value = SimpleNamespace(
            response=AssistantMessage("answer")
        )

        main()

    config = client_class.call_args.kwargs["config_provider"]
    assert config.model == "qwen3.8-27b"

    messages = client.call.call_args.kwargs["messages"]
    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert messages[0].get_message() == "You are a helpful assistant."
    assert isinstance(messages[1], UserMessage)
    assert messages[1].get_message() == "Hello, how are you?"
    assert capsys.readouterr().out == "Response: answer\n"
