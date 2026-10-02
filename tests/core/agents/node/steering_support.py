from concurrent.futures import Future
from threading import Event, Thread

from exmekh_lib.core.agents.deep_agent.agent_response import AgentResponse
from exmekh_lib.core.agents.deep_agent.deep_agent import DeepAgent
from exmekh_lib.core.models.assistant_message import (
    AssistantMessage,
    FunctionCall,
    ToolCall,
)
from exmekh_lib.core.models.user_message import UserMessage


class AgentRuns:
    """Launch test runs and release their gates before joining during cleanup."""

    def __init__(self):
        self._threads: list[Thread] = []
        self._releases: list[Event] = []

    def __call__(self, agent: DeepAgent, message="start") -> Future[AgentResponse]:
        result: Future[AgentResponse] = Future()

        def run():
            try:
                result.set_result(agent.run(message))
            except BaseException as error:
                result.set_exception(error)

        thread = Thread(target=run, daemon=True)
        self._threads.append(thread)
        thread.start()
        return result

    def release_on_cleanup(self, event: Event) -> None:
        self._releases.append(event)

    def close(self) -> None:
        for release in self._releases:
            release.set()
        for thread in self._threads:
            thread.join(5)
            assert not thread.is_alive()


def tool_reply(*calls) -> AssistantMessage:
    return AssistantMessage(None, tool_calls=list(calls))


def call(name="mock-tool", arguments=None, call_id="origin") -> ToolCall:
    return ToolCall(call_id, FunctionCall(name, arguments or {}))


def user_contents(request) -> list:
    return [
        message.get_message()
        for message in request.get_messages()
        if isinstance(message, UserMessage)
    ]


def blocked_work(gates, launch_run):
    started = Event()
    release = gates()
    launch_run.release_on_cleanup(release)

    def work():
        started.set()
        assert release.wait(5)
        return "finished"

    return work, started, release
