# DeepAgent & Hierarchical Execution

The [`DeepAgent`](file:///c:/Users/danie/Desktop/qs/qs/src/qs/core/agents/deep_agent/deep_agent.py) engine orchestrates multi-depth skill execution, tool calling loops, real-time streaming, and hierarchical history tree management.

---

## 1. Initializing & Executing a DeepAgent

To run a `DeepAgent`, instantiate a root [`Node`](file:///c:/Users/danie/Desktop/qs/qs/src/qs/core/agents/node/node.py) pointing to a capabilities directory, bind a workspace path (`work_root`), and supply an LLM provider:

```python
from pathlib import Path
from qs.core.agents.deep_agent import DeepAgent
from qs.core.agents.node import Node
from qs.providers.local import LocalProvider

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
)

response = agent.run("Analyze the codebase and run tests.")
print("Response:", response.get_message())
```

---

## 2. Skill Navigation & Depth Tracking

As the agent executes, it transitions between skills based on the active HDS and skill catalog definitions:

| Active Scope | `skill_path` | `depth` |
|---|---|---|
| Main HDS System Prompt | `/` | `0` |
| Top-level Skill | `/mock-skill` | `1` |
| Nested Sub-Skill | `/mock-skill/nested-skill` | `2` |

- **Skill Entry**: Triggered when the agent calls a skill tool.
- **Exit Instructions**: The selected exit tool provides plain Markdown instructions, appended after each skill catalog. They have no separate callable capability.
- **Skill Exit**: The HDS-selected exit tool is allowed only at positive depth and returns the current skill result to its caller.

---

## 3. History vs. History Tree

The [`AgentResponse`](file:///c:/Users/danie/Desktop/qs/qs/src/qs/core/agents/deep_agent/agent_response.py) returned by `agent.run(...)` maintains two complementary representations of conversation history:

1. **`response.history`**: A linear list of standard message dictionaries representing the top-level chat sequence.
2. **`response.get_history_tree()`**: A nested snapshot tree reflecting skill activations, sub-agent scopes, and tool execution states.

```python
response = agent.run("Perform refactoring task")

# Linear history list
linear_history = response.history

# Hierarchical skill history tree
tree_history = response.get_history_tree()
```

---

## 4. Real-time Event Streaming (`on_stream`)

Pass an `on_stream` callback to `DeepAgent` to receive real-time updates for both LLM output generation and tool execution states:

```python
from qs.core.agents.deep_agent import (
    DeepAgentEvent,
    DeepAgentStreamEvent,
    DeepAgentExecutionEvent,
)
from qs.core.models.provider_stream_event import ProviderStreamEventType


def handle_stream(event: DeepAgentEvent) -> None:
    if isinstance(event, DeepAgentStreamEvent):
        # Tokens streamed directly from provider
        provider_event = event.provider_event
        if provider_event.type is ProviderStreamEventType.RESPONSE:
            print(provider_event.content, end="", flush=True)

    elif isinstance(event, DeepAgentExecutionEvent):
        # Tool execution start / end events with full context
        print(
            f"\n[Tool Execution]: {event.tool_name} (type={event.type}, depth={event.depth}, path={event.skill_path})"
        )


agent = DeepAgent(
    provider=provider,
    node=root_node,
    work_root=Path.cwd(),
    on_stream=handle_stream,
)
```

### Event Classification:
- **[`DeepAgentStreamEvent`](file:///c:/Users/danie/Desktop/qs/qs/src/qs/core/agents/deep_agent/stream_event.py)**: Low-level provider token streaming (`START`, `REASONING`, `RESPONSE`, `END`), tagged with `depth` and `skill_path`.
- **[`DeepAgentExecutionEvent`](file:///c:/Users/danie/Desktop/qs/qs/src/qs/core/agents/deep_agent/stream_event.py)**: Emitted when tools start (`TOOL_START`) or complete (`TOOL_END`). Includes snapshots of linear `history` and hierarchical `history_tree`.

---

## 5. Workspace Isolation (`work_root`)

`work_root` MUST be a valid, readable, and writable directory on disk:
- It is validated by `DeepAgent._validate_work_root(work_root)`.
- Automatically passed as the positional `work_root: Path` parameter to any function decorated with `@deep_tool`.
- Supplies the tools' working directory. Each tool must enforce any filesystem
  boundaries it requires; passing `work_root` is not an operating-system sandbox.

## 6. Runtime and Saved Tool Outputs

`Node.initialize()` composes a `NodeRuntimeConfig` and a `NodeRuntime`.
The configuration builds the client tools, system prompt, executor and node
cache. `ToolCallExecutor` runs a resolved call and returns a
`ToolExecutionResult`; the runtime applies its history update, saves eligible
results, and emits execution events.

Ordinary tools and exit tools produce cache records. Deep skill instructions,
including the selected exit tool instructions, remain conversational messages. Records capture arguments before
execution, the active skill path before an exit, the function result, and the
provider's thinking from the calling turn. This keeps the result accessible
after skill history compaction. See [Tool output caching](09-tool-call-cache.md)
for executable examples and retrieval APIs.

`persistent_history=False` rebuilds the client and execution state before each
run; it does not erase saved outputs. Each runtime has its own generated node ID,
including when several nodes share the same HDS and tokenizer session.

### Multiple calls in one provider response

The runtime validates all requested capability names against the scope that
generated the response before executing the batch. Ordinary tools execute first
in that scope, followed by deep skills in their requested order, then exit tools.
Opening a deep skill suspends sibling skills and exits until execution returns
to the original depth; deferred calls retain their original turn's thinking.
Deferred calls appear in a separate assistant message when resumed, so provider
continuations never contain unanswered sibling call IDs. History compaction
preserves ordinary tool responses from the opening batch.

The session cache synchronizes storage operations across threads. It does not
make concurrent `run()` calls on the same runtime safe.
