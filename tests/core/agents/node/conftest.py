from contextlib import ExitStack
from pathlib import Path
from threading import Event

import pytest

from exmekh_lib.core.agents.capabilities.defaults import DefaultCapabilities
from exmekh_lib.core.agents.deep_agent.deep_agent import DeepAgent
from exmekh_lib.core.agents.node import Node
from exmekh_lib.core.cache import SessionCache
from tests.core.agents.node.steering_support import AgentRuns
from tests.core.agents.node.test_node import _write_nested_capabilities


@pytest.fixture
def gates():
    events = []

    def create():
        event = Event()
        events.append(event)
        return event

    yield create
    for event in events:
        event.set()


@pytest.fixture
def make_agent(tmp_path: Path):
    with ExitStack() as cleanup:
        agents = []

        def build(
            provider, *tools, persistent=True, on_stream=None, children=(), include=()
        ):
            directory = tmp_path / f"agent-{len(agents)}"
            path = _write_nested_capabilities(directory)
            session = cleanup.enter_context(SessionCache(directory / "cache"))
            provider.get_session = lambda: session
            counter = iter(range(100))
            provider.generate_id = lambda: f"output-{next(counter)}"
            node = Node(
                [path],
                name_hds="main",
                persistent_history=persistent,
                children=children,
                default_capabilities=DefaultCapabilities(
                    harness=False,
                    include={
                        "read_tool_output",
                        "control_tool_execution",
                        "call_sub_agent",
                        *include,
                    },
                    overrides={tool.name: tool for tool in tools},
                ),
            )
            agent = DeepAgent(provider, node, directory, on_stream=on_stream)
            agents.append(agent)
            return agent

        yield build


@pytest.fixture
def launch_run():
    runs = AgentRuns()
    yield runs
    runs.close()
