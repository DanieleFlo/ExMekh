# qs

## Deep-agent streaming

Pass an `on_stream` callback to `DeepAgent` to enable provider streaming. The
callback receives lightweight provider events separately from contextual tool
and skill execution events. Provider events include only the current depth and
skill path; execution events also include history and history-tree snapshots.

```python
from qs.core.agents.deep_agent import (
    DeepAgentEvent,
    DeepAgentExecutionEvent,
    DeepAgentStreamEvent,
)
from qs.core.agents.deep_agent.deep_agent import DeepAgent
from qs.core.models.provider_stream_event import ProviderStreamEventType


def handle_stream(event: DeepAgentEvent) -> None:
    if isinstance(event, DeepAgentStreamEvent):
        provider_event = event.provider_event
        if provider_event.type is not ProviderStreamEventType.RESPONSE:
            return
        send_token(
            content=provider_event.content,
            depth=event.depth,
            skill_path=event.skill_path,
        )
        return

    if isinstance(event, DeepAgentExecutionEvent):
        send_tool_state(
            name=event.tool_name,
            call_id=event.tool_call_id,
            is_running=event.is_tool_running,
            depth=event.depth,
            skill_path=event.skill_path,
            history=event.history,
            history_tree=event.history_tree,
        )


agent = DeepAgent(provider=provider, node=root_node, on_stream=handle_stream)
```

`skill_path` is `/` while the HDS is active, `/mock-skill` inside a root skill,
and `/mock-skill/nested-skill` inside a nested skill. History snapshots are not
copied for each generated token. Tool end events are emitted even when tool
execution raises an exception.

## Function tools

Register typed Python functions with `Tool` and include the registry in a
`ClientRequest`. The OpenAI payload schema is generated from the function
annotations and its docstring.

```python
from typing import Literal

from qs.core.models.client import Client
from qs.core.models.client_request import ClientRequest
from qs.core.models.system_message import SystemMessage
from qs.core.models.tool import Tool
from qs.core.models.tools import Tools
from qs.core.models.user_message import UserMessage
from qs.core.providers.openai_tool_mapper import OpenAIToolMapper


def weather(city: str, units: Literal["celsius", "fahrenheit"] = "celsius") -> str:
    """Return the weather for a city."""
    return f"Weather for {city} in {units}"


tools = Tools([Tool(weather)])
client = Client(
    provider=provider,
    system_message=SystemMessage("Use tools when needed."),
)
request = ClientRequest(
    UserMessage("Weather in Rome?"),
    tools=tools,
    tool_choice="auto",
)
response = client.call(request)

assistant_message = response.response
for tool_call in assistant_message.get_tool_calls() or []:
    if tool_call["type"] == "function":
        selected_tool, arguments = OpenAIToolMapper.resolve_tool_call(
            tool_call,
            tools,
        )
        result = selected_tool.tool(**arguments)
```

Parameters must be named and annotated. The mapper supports JSON primitives,
unions and optional values, `Literal`, enums, lists, sets, tuples, mappings,
`TypedDict`, dataclasses, `Annotated`, `NewType`, dates, UUIDs, decimals, and
paths. Positional-only and variadic parameters are rejected because OpenAI
function arguments are JSON objects.

## Structured responses

Use `call_structured` with a Pydantic model type when the assistant response
must be a validated object instead of a string. Strict schema generation and
strict Pydantic validation are enabled by default.

```python
from pydantic import BaseModel, ValidationError

from qs.core.models.client_request import ClientRequest
from qs.core.models.user_message import UserMessage


class Answer(BaseModel):
    summary: str
    confidence: float


try:
    response = client.call_structured(
        ClientRequest(UserMessage("Summarize the document.")),
        Answer,
    )
    answer = response.response.get_message()
except ValidationError as error:
    print(error.errors())
```

Pass `strict=False` to allow Pydantic coercion while still validating the
returned JSON against the model. Invalid JSON and incompatible values raise
the original `ValidationError`, including every field location and error
detail.

## Live tests

Live tests use `BASE_URL` and `API_KEY` from the environment or `.env`. They
target `qwen3.5-9b-mtp` by default; set `QS_LIVE_MODEL` to use another model.
The normal `uv run pytest` command collects these tests but skips them.

Opt in with the pytest boolean flag:

```shell
uv run pytest --run-live tests/live
```

Alternatively, enable them through an environment variable:

```shell
QS_RUN_LIVE_TESTS=1 uv run pytest tests/live
```

Live assertion messages start with `CLIENT` for protocol or history failures
and `MODEL` for failures in the requested model behavior.
