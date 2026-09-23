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
    children=[
        Node(
            capabilities_paths=[capabilities_path],
            name_hds="mock-sub-agent",
            persistent_history=False,
        )
    ],
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

## 3. Sub-agent Delegation

A node with at least one direct child exposes this runtime tool by default:

```text
call_sub_agent(sub_agent: <direct-child enum>, text: <non-empty string>)
```

`sub_agent` is constrained in the generated JSON Schema to the HDS names of the
node's direct children. `text` is required and has both a schema `minLength` of
one and a non-whitespace pattern; runtime validation also rejects whitespace-only values. The tool is available
from the HDS and from every active skill or sub-skill, including skills that do
not inherit parent tools. A leaf node does not expose it.
The tool description lists the direct child names and includes an XML
`<sub_agents>` block with each child's HDS name and description.
With `DefaultCapabilities`, a node can exclude or override `call_sub_agent`;
an override must be a `Tool` named `call_sub_agent`. Selection is independent
for each parent and child. See [per-node defaults](04-capabilities-and-catalog.md#per-node-default-capabilities).

Calling the tool runs `child.run(text)`, so the child receives the value exactly
as a new user message. Repeated calls follow the child's `persistent_history`
setting. A grandchild is not selectable directly: its parent must delegate to it
in a subsequent `call_sub_agent` call.

The child final response becomes the parent tool result. It receives the normal
XML tool header and is saved in the parent's node-scoped cache with the calling
skill path. The child's full conversation, internal tool calls and saved outputs
remain isolated in the child runtime. Input errors are returned as tool results
so the parent model can retry; provider and child execution failures still
propagate as exceptions.

Sibling child HDS names must be unique. `call_sub_agent` is a reserved runtime
tool name on non-leaf nodes and cannot also be declared as a filesystem tool.

### Agent Context ID & Engine Delegation
Inference engines such as DS4 require a non-empty `agent_context_id` in the `agentic` payload. Each `Node` can define its own `agent_context_id` (via Markdown frontmatter `agent-context-id`, falling back to `name_hds`, or via `Node(..., agent_context_id=...)`). During sub-agent execution:
- The parent node makes requests using its parent `agent_context_id`.
- When delegating via `call_sub_agent`, the child node executes under its own distinct `agent_context_id`.
- DS4 observes the context switch, suspends the parent's pending tool call state, and saves a temporary `.dsh` checkpoint.
- Upon child completion and return, the parent resumes with its original `agent_context_id`, causing DS4 to restore the parent's checkpoint and prefill only the returned tool output.

All distinct nodes within a `DeepAgent` hierarchy must have unique `agent_context_id` values to prevent context collisions.

---

## 4. History vs. History Tree

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

## 5. Real-time Event Streaming (`on_stream`)

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

## 6. Workspace Isolation (`work_root`)

`work_root` MUST be a valid, readable, and writable directory on disk:
- It is validated by `DeepAgent._validate_work_root(work_root)`.
- Automatically passed as the positional `work_root: Path` parameter to any function decorated with `@deep_tool`.
- Supplies the tools' working directory. Each tool must enforce any filesystem
  boundaries it requires; passing `work_root` is not an operating-system sandbox.

## 7. Runtime and Saved Tool Outputs

`Node.initialize()` composes a `NodeRuntimeConfig` and a `NodeRuntime`.
The configuration builds the client tools, system prompt, executor and node
cache. `ToolCallExecutor` runs a resolved call and returns a
`ToolExecutionResult`; the runtime applies its history update, saves eligible
results, and emits execution events.

Ordinary tools produce cache records. When an exit tool closes a deep skill, the
record is attributed to the opening skill call and retains its name, arguments,
call ID and opening-turn thinking. Deep skill instructions remain conversational
messages, while the final output stays accessible after history compaction.
The final output's XML header also includes the exit call's thinking when present.
Its excerpt is limited to the first 100 model tokens. The cached opening-turn
thinking remains complete.

The runtime automatically exposes `get_execution_history`,
`search_saved_tool_outputs` and `read_saved_tool_output` in every scope. They
return XML, operate only on the current node and do not cache their own results.
See [Tool output caching](09-tool-call-cache.md) for formats and lifecycle.

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
