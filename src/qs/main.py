import json
import logging
from pathlib import Path

from dotenv import load_dotenv

from qs.config.capabilities import CAPABILITY_FORMAT
from qs.config.cli import CliPresentation, CliSettings
from qs.config.provider import ProviderSettings
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


def create_stream_printer(
    presentation: CliPresentation | None = None,
) -> DeepAgentEventHandler:
    presentation = presentation or CliPresentation()
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
                    presentation.thinking_label
                    if provider_event.type is ProviderStreamEventType.REASONING
                    else presentation.response_stream_label
                )
                try:
                    print(f"\n{label}: ", end="", flush=True)
                except (OSError, ValueError) as e:
                    logger.error(e)

                current_section = provider_event.type
            try:
                print(provider_event.content, end="", flush=True)
            except (OSError, ValueError) as e:
                logger.error(e)
            return

        if isinstance(event, DeepAgentExecutionEvent):
            current_section = None
            print(
                f"\n{event.type}: {event.tool_name} "
                f"(depth={event.depth}, path={event.skill_path})"
            )

    return print_stream_event


def main() -> None:
    cli_settings = CliSettings.from_environment()
    logging.basicConfig(
        level=logging.INFO,
        format=cli_settings.presentation.log_format,
    )
    provider_settings = ProviderSettings.from_environment()
    provider = LocalProvider(model=provider_settings.model, api=provider_settings.api)
    capabilities_paths = [Path(__file__).parent / CAPABILITY_FORMAT.root_folder]
    root_node = Node(
        capabilities_paths=capabilities_paths,
        name_hds=cli_settings.root_hds,
        children=[
            Node(
                capabilities_paths=capabilities_paths,
                name_hds=cli_settings.child_hds,
            )
        ],
    )
    deep_agent = DeepAgent(
        provider=provider,
        node=root_node,
        work_root=Path.cwd(),
        on_stream=create_stream_printer(cli_settings.presentation),
    )
    while True:
        user_message = input(cli_settings.presentation.user_prompt)
        if user_message.lower() in cli_settings.presentation.exit_commands:
            print(cli_settings.presentation.exit_message)
            break
        response = deep_agent.run(user_message=user_message)
        print(cli_settings.presentation.response_label, response.get_message())
        print(
            cli_settings.presentation.history_label,
            json.dumps(response.history, indent=2, ensure_ascii=True),
        )
        print(
            cli_settings.presentation.skill_history_label,
            json.dumps(response.get_history_tree(), indent=2, ensure_ascii=True),
        )


if __name__ == "__main__":
    main()
