# `qs` — Hierarchical LLM Agent Framework

`qs` is a modular, strongly-typed Python framework for building structured, reliable, and hierarchical Large Language Model (LLM) agents. It combines dynamic function tool calling, Pydantic structured outputs, filesystem-driven capabilities (HDS system prompts, skills, and tools), and multi-depth agent execution with real-time streaming and history tree snapshotting.

---

## Key Features

- **Hierarchical Agent Engine ([`DeepAgent`](file:///c:/Users/danie/Desktop/qs/qs/docs/05-deep-agent-and-execution.md))**: Execute multi-depth, scoped agent hierarchies with dynamic skill tree navigation, history snapshotting, and cycle detection.
- **Filesystem Capabilities ([`Capabilities`](file:///c:/Users/danie/Desktop/qs/qs/docs/04-capabilities-and-catalog.md))**: Define system prompts (HDS), skills, exit skills, and Python function tools directly in Markdown and Python files inside a modular `capabilities/` directory.
- **Typed Function Tools & Auto Schema ([`Tool`](file:///c:/Users/danie/Desktop/qs/qs/docs/03-tools-and-structured-outputs.md))**: Register annotated Python functions. Automatic JSON schema generation, type hint reflection, and value coercion (handling dates, UUIDs, enums, literals, dataclasses, and Pydantic models).
- **Structured Outputs**: Validate model completions directly into Pydantic models via `client.call_structured(...)` with optional strict schema enforcement.
- **Real-Time Token & Event Streaming**: Stream raw provider tokens (including reasoning/thinking tokens) alongside structured execution events (`tool_start`, `tool_end`, depth, and skill path).
- **Flexible Provider Backend ([`LocalProvider`](file:///c:/Users/danie/Desktop/qs/qs/docs/06-providers-and-configuration.md))**: Native integration with local LLM servers (llama.cpp, Ollama, vLLM, LM Studio) and OpenAI-compatible Chat Completions or Responses endpoints.

---

## Documentation Index

Explore the detailed documentation in the [`docs/`](file:///c:/Users/danie/Desktop/qs/qs/docs) directory:

1. **[Architecture & Core Concepts](file:///c:/Users/danie/Desktop/qs/qs/docs/01-architecture-and-concepts.md)**: High-level design, core building blocks, and system lifecycle.
2. **[Client & Messaging Model](file:///c:/Users/danie/Desktop/qs/qs/docs/02-client-and-models.md)**: `Client` API, message data structures (`UserMessage`, `AssistantMessage`, `SystemMessage`, `ToolMessage`), and history rules.
3. **[Function Tools & Structured Outputs](file:///c:/Users/danie/Desktop/qs/qs/docs/03-tools-and-structured-outputs.md)**: Function tool registration, type coercion, `@deep_tool` binding, and Pydantic validation.
4. **[Capabilities System & Catalog](file:///c:/Users/danie/Desktop/qs/qs/docs/04-capabilities-and-catalog.md)**: Directory layout, HDS prompts, Markdown skills, exit tools, and dynamic catalog attachment.
5. **[DeepAgent & Hierarchical Execution](file:///c:/Users/danie/Desktop/qs/qs/docs/05-deep-agent-and-execution.md)**: Multi-depth execution, skill path tracking (`/`, `/skill-a`), linear vs. tree history, and workspace root isolation.
6. **[Providers & Configuration](file:///c:/Users/danie/Desktop/qs/qs/docs/06-providers-and-configuration.md)**: `LocalProvider` configuration, Chat Completions vs. Responses APIs, and `.env` setup.
7. **[CLI, Testing & Live Verification](file:///c:/Users/danie/Desktop/qs/qs/docs/07-cli-and-testing.md)**: Interactive REPL CLI (`qs`), running offline `pytest` suites, and running live model tests.

---

## Installation & Requirements

### Requirements
- Python `>= 3.13`
- [`uv`](https://github.com/astral-sh/uv) (recommended package manager)

### Installation
Clone the repository and sync project dependencies using `uv`:

```shell
git clone <repository-url>
cd qs
uv sync
```

---

## Quickstart

### 1. Simple Client & Function Tool Calling

```python
from typing import Literal
from qs.core.models.client import Client
from qs.core.models.client_request import ClientRequest
from qs.core.models.system_message import SystemMessage
from qs.core.models.tool import Tool
from qs.core.models.tools import Tools
from qs.core.models.user_message import UserMessage
from qs.providers.local import LocalProvider


# Define a typed python tool
def get_weather(city: str, units: Literal["celsius", "fahrenheit"] = "celsius") -> str:
    """Return weather information for a specified city."""
    return f"Weather in {city}: 22° {units}"


# Instantiate provider and client
provider = LocalProvider(model="Qwen3.8-27B-UD-Q4_K_S")
client = Client(
    provider=provider,
    system_message=SystemMessage("You are a helpful assistant."),
    tools=Tools([Tool(get_weather)]),
)

# Make request
response = client.call(ClientRequest(UserMessage("What is the weather in Rome?")))
print("Assistant:", response.get_message())
```

---

### 2. Hierarchical `DeepAgent` with Real-Time Event Streaming

```python
from pathlib import Path
from qs.core.agents.deep_agent import (
    DeepAgent,
    DeepAgentEvent,
    DeepAgentExecutionEvent,
    DeepAgentStreamEvent,
)
from qs.core.agents.node import Node
from qs.core.models.provider_stream_event import ProviderStreamEventType
from qs.providers.local import LocalProvider


def handle_stream_events(event: DeepAgentEvent) -> None:
    if isinstance(event, DeepAgentStreamEvent):
        if event.provider_event.type is ProviderStreamEventType.RESPONSE:
            print(event.provider_event.content, end="", flush=True)
    elif isinstance(event, DeepAgentExecutionEvent):
        print(
            f"\n[Tool Execution]: {event.tool_name} (depth={event.depth}, path={event.skill_path})"
        )


provider = LocalProvider(model="Qwen3.8-27B-UD-Q4_K_S")
capabilities_path = Path.cwd() / "src" / "qs" / "capabilities"

root_node = Node(
    capabilities_paths=[capabilities_path],
    name_hds="main",
)

agent = DeepAgent(
    provider=provider,
    node=root_node,
    work_root=Path.cwd(),
    on_stream=handle_stream_events,
)

response = agent.run("Perform workspace maintenance and check capabilities.")
```

---

### 3. Structured Output Parsing (Pydantic)

```python
from pydantic import BaseModel
from qs.core.models.client import Client
from qs.core.models.client_request import ClientRequest
from qs.core.models.system_message import SystemMessage
from qs.core.models.user_message import UserMessage


class TaskSummary(BaseModel):
    title: str
    completed: bool
    estimated_hours: float


client = Client(provider=provider, system_message=SystemMessage("Extract task info."))
response = client.call_structured(
    ClientRequest(
        UserMessage("Analyze task: Fix login bug, 3.5 hours estimated, status done")
    ),
    TaskSummary,
)

summary: TaskSummary = response.response.get_message()
print(f"Task '{summary.title}' - Completed: {summary.completed}")
```

---

## Running Tests

### Unit Tests
Run offline unit test suite:

```shell
uv run pytest
```

### Live Provider Integration Tests
Opt-in to contact a live model provider endpoint (requires `BASE_URL` set in `.env`):

```shell
uv run pytest --run-live tests/live
```
