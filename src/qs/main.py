from dotenv import load_dotenv

from qs.config.local_provider import LocalProviderConfig
from qs.core.models.client import Client
from qs.core.models.system_message import SystemMessage
from qs.core.models.user_message import UserMessage

load_dotenv()


def main():
    model = "qwen3.8-27b"
    provider_config = LocalProviderConfig(model=model)
    client = Client(config_provider=provider_config)

    system_message = SystemMessage(content="You are a helpful assistant.")
    user_message = UserMessage(content="Hello, how are you?")
    messages = [system_message, user_message]

    response = client.call(messages=messages)
    print("Response:", response.response)


if __name__ == "__main__":
    main()
