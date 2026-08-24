import logging
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel

from qs.core.models.client import Client
from qs.core.models.client_request import ClientRequest
from qs.core.models.reasoning_effort import ReasoningEffort
from qs.core.models.system_message import SystemMessage
from qs.core.models.tool_message import ToolMessage
from qs.core.models.tools import Tool, Tools
from qs.core.models.user_message import UserMessage
from qs.providers.local import LocalProvider

load_dotenv()
logger = logging.getLogger(__name__)


def main(
    model: str = "qwen3.5-9b-mtp",
    system_prompt: str = "You are a helpful assistant.",
    user_prompt: str = "Hello, how are you?",
    reasoning_effort: ReasoningEffort | None = None,
) -> None:
    def weather(city: str, units: Literal["celsius", "fahrenheit"] = "celsius") -> str:
        """Return the weather for a city."""
        return f"Weather for {city} in {units}"

    tools = Tools([Tool(weather)])
    provider = LocalProvider(model=model, reasoning_effort=reasoning_effort)

    system_message = SystemMessage(content=system_prompt)
    client = Client(provider=provider, system_message=system_message, tools=tools)

    user_message = UserMessage(content=user_prompt)
    request = ClientRequest(message=user_message)

    response = client.call(request=request)
    print("S---S")
    print("Response:", response.response.get_message())
    print("---")
    print("Tokens used:", response.usage)
    print("---")
    print("Thinking:", response.thinking)
    print("---")
    print("Message history", client.dump_history())
    print("E---E")

    user_message = UserMessage(content="Weather in Rome?")
    request = ClientRequest(message=user_message)

    response = client.call(request=request)
    tool_calls = response.response.get_tool_calls()
    print("S---S")
    print("Response:", response.response.get_message())
    print("---")
    print("Tool calls:", tool_calls)
    print("---")
    print("Tokens used:", response.usage)
    print("---")
    print("Thinking:", response.thinking)
    print("---")
    print("Message history", client.dump_history())
    print("E---E")

    if tool_calls and tool_calls[0]:
        tool_id = tool_calls[0].id
        tool_response = ToolMessage(
            content='{"city":"Rome","temperature":"28 °C"}', tool_call_id=tool_id
        )
        client.add_tool_response(tool_response)
        response = client.continue_after_tools()

        print("S---S")
        print("Tool response added to history.")
        print("Response:", response.response.get_message())
        print("Message history", client.dump_history())
        print("E---E")

        response = client.call(request=request)

        class TempModel(BaseModel):
            temperature: str
        ser_message = UserMessage(content="Quale temperatura c'è a Roma?")
        request = ClientRequest(message=ser_message, tool_choice="none")
        response = client.call_structured(request=request, response_model=TempModel)
        print("S---S")
        print("Response:", response.response.get_message())
        print("---")
        print("Tokens used:", response.usage)
        print("---")
        print("Thinking:", response.thinking)
        print("---")
        print("Message history", client.dump_history())
        print("E---E")


if __name__ == "__main__":
    main()
