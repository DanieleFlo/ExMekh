import logging

from dotenv import load_dotenv

from qs.config.local_provider import LocalProviderConfig, ReasoningEffort
from qs.core.models.client import Client
from qs.core.models.system_message import SystemMessage
from qs.core.models.user_message import UserMessage

load_dotenv()
logger = logging.getLogger(__name__)


def main(
    model: str = "qwen3.8-27b",
    system_prompt: str = "You are a helpful assistant.",
    user_prompt: str = "Hello, how are you?",
    reasoning_effort: ReasoningEffort | None = ReasoningEffort.XHIGH,
) -> None:
    provider_config = LocalProviderConfig(
        model=model, reasoning_effort=reasoning_effort
    )
    client = Client(config_provider=provider_config)

    system_message = SystemMessage(content=system_prompt)
    user_message = UserMessage(content=user_prompt)
    messages = [system_message, user_message]

    response = client.call(messages=messages)
    print("Response:", response.response.get_message())
    print("Tokens used:", response.usage)
    print("Thinking:", response.thinking)
    print("Message history", client.dump_history())


if __name__ == "__main__":
    main()
