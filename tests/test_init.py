from unittest.mock import patch

import pytest

from qs.config.local_provider import ReasoningEffort
from qs.core.models.assistant_message import AssistantMessage
from qs.core.models.model_thinking import ModelThinking
from qs.core.models.model_usage import ModelUsage
from qs.core.models.provider_response import ProviderResponse
from qs.core.models.system_message import SystemMessage
from qs.core.models.user_message import UserMessage
from qs.main import main


@pytest.mark.parametrize(
    ("model", "system_prompt", "user_prompt", "effort"),
    [
        (
            "custom-model-a",
            "Follow policy alpha.",
            "Question with a newline\nand symbols <>&",
            ReasoningEffort.LOW,
        ),
        (
            "custom-model-b:revision-2",
            "Follow a different policy.",
            "A completely different request",
            None,
        ),
    ],
)
def test_main_sends_explicit_messages_and_prints_complete_response(
    capsys,
    model: str,
    system_prompt: str,
    user_prompt: str,
    effort: ReasoningEffort | None,
) -> None:
    usage = ModelUsage(17, 23, 40)
    thinking = ModelThinking("non-default reasoning")
    history_dump = [{"role": "system", "content": system_prompt}]

    with patch("qs.main.Client") as client_class:
        client = client_class.return_value
        client.call.return_value = ProviderResponse(
            AssistantMessage("custom answer"), thinking=thinking, usage=usage
        )
        client.dump_history.return_value = history_dump

        main(model, system_prompt, user_prompt, effort)

    config = client_class.call_args.kwargs["config_provider"]
    assert config.model == model
    assert config.reasoning_effort is effort

    messages = client.call.call_args.kwargs["messages"]
    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert messages[0].get_message() == system_prompt
    assert isinstance(messages[1], UserMessage)
    assert messages[1].get_message() == user_prompt
    assert capsys.readouterr().out.splitlines() == [
        "Response: custom answer",
        f"Tokens used: {usage}",
        f"Thinking: {thinking}",
        f"Message history {history_dump}",
    ]
