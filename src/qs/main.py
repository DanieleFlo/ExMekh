import json
import logging
from pathlib import Path

from dotenv import load_dotenv

from qs.core.agents.deep_agent import (
    DeepAgentEvent,
    DeepAgentEventHandler,
    DeepAgentExecutionEvent,
    DeepAgentStreamEvent,
)
from qs.core.agents.deep_agent.deep_agent import DeepAgent
from qs.core.agents.node import Node
from qs.core.models.provider_stream_event import ProviderStreamEventType
from qs.providers.local import LocalProvider

load_dotenv()
logger = logging.getLogger(__name__)


def create_stream_printer() -> DeepAgentEventHandler:
    current_section: ProviderStreamEventType | None = None

    def print_stream_event(event: DeepAgentEvent) -> None:
        nonlocal current_section

        if isinstance(event, DeepAgentStreamEvent):
            provider_event = event.provider_event
            if provider_event.type is ProviderStreamEventType.START:
                current_section = None
                return
            if provider_event.type is ProviderStreamEventType.END:
                if current_section is not None:
                    print()
                current_section = None
                return
            if (
                provider_event.type
                not in {
                    ProviderStreamEventType.REASONING,
                    ProviderStreamEventType.RESPONSE,
                }
                or provider_event.content is None
            ):
                return

            if provider_event.type is not current_section:
                label = (
                    "Thinking"
                    if provider_event.type is ProviderStreamEventType.REASONING
                    else "Response"
                )
                print(f"\n{label}: ", end="", flush=True)
                current_section = provider_event.type
            print(provider_event.content, end="", flush=True)
            return

        if isinstance(event, DeepAgentExecutionEvent):
            current_section = None
            print(
                f"\n{event.type}: {event.tool_name} "
                f"(depth={event.depth}, path={event.skill_path})"
            )

    return print_stream_event


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    provider = LocalProvider(model="qwen3.5-9b-mtp")
    capabilities_paths = [Path(__file__).parent / "capabilities"]
    root_node = Node(
        capabilities_paths=capabilities_paths,
        name_hds="main",
    )
    deep_agent = DeepAgent(
        provider=provider,
        node=root_node,
        on_stream=create_stream_printer(),
    )
    while True:
        user_message = input("User: ")
        if user_message.lower() in ["exit", "quit"]:
            print("Exiting...")
            break
        response = deep_agent.run(user_message=user_message)
        print("\nResponse:", response.get_message())
        print("history:", json.dumps(response.history, indent=2, ensure_ascii=True))
        print(
            "Deep skill history:\n",
            json.dumps(response.get_history_tree(), indent=2, ensure_ascii=True),
        )


if __name__ == "__main__":
    main()
