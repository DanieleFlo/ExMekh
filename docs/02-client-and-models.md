# Client & Messaging Model

The `qs` framework provides a strongly-typed, object-oriented layer over raw LLM calls. The primary interaction point is the [`Client`](file:///c:/Users/danie/Desktop/qs/qs/src/qs/core/models/client.py) class, which maintains conversation history, manages system prompts, and handles tool responses and event streaming.

---

## Messaging Hierarchy & Data Models

All chat messages in `qs` implement the [`Message`](file:///c:/Users/danie/Desktop/qs/qs/src/qs/interfaces/message.py) interface.

```
                  +-------------------+
                  |     Message       |
                  +-------------------+
                            |
     +----------------------+----------------------+----------------------+
     |                      |                      |                      |
+----+----+            +----+----+            +----+----+            +----+----+
|UserMessage|          |SystemMessage|        |Assistant|            |ToolMessage|
+---------+            +---------+            | Message |            +---------+
                                              +----+----+
                                                   |
                                            +------+------+
                                            |Structured   |
                                            |AssistantMsg |
                                            +-------------+
```

### Core Message Types

1. **[`SystemMessage`](file:///c:/Users/danie/Desktop/qs/qs/src/qs/core/models/system_message.py)**: Sets agent role or instructions. Must be the first message in conversation history.
2. **[`UserMessage`](file:///c:/Users/danie/Desktop/qs/qs/src/qs/core/models/user_message.py)**: Represents input from the user or prompt source. Supports plain text and structured content payloads.
3. **[`AssistantMessage`](file:///c:/Users/danie/Desktop/qs/qs/src/qs/core/models/assistant_message.py)**: Output emitted by the LLM. Contains:
   - `content`: Text response string (if any).
   - `thinking`: Optional [`ModelThinking`](file:///c:/Users/danie/Desktop/qs/qs/src/qs/core/models/model_thinking.py) containing reasoning/chain-of-thought tokens.
   - `tool_calls`: List of tool call specifications requested by the model.
   - `usage`: Optional [`ModelUsage`](file:///c:/Users/danie/Desktop/qs/qs/src/qs/core/models/model_usage.py) tracking prompt/completion tokens.
4. **[`ToolMessage`](file:///c:/Users/danie/Desktop/qs/qs/src/qs/core/models/tool_message.py)**: Contains the return output of a tool call, paired with `tool_call_id` and `tool_name`.
5. **[`FunctionMessage`](file:///c:/Users/danie/Desktop/qs/qs/src/qs/core/models/function_message.py)**: Legacy function result message.
6. **[`StructuredAssistantMessage`](file:///c:/Users/danie/Desktop/qs/qs/src/qs/core/models/structured_assistant_message.py)**: Wraps model responses when validated against a Pydantic schema via `call_structured`.

---

## Client API Usage

### 1. Basic Synchronous Calls

```python
from qs.core.models.client import Client
from qs.core.models.client_request import ClientRequest
from qs.core.models.system_message import SystemMessage
from qs.core.models.user_message import UserMessage
from qs.providers.local import LocalProvider

provider = LocalProvider(model="Qwen3.8-27B-UD-Q4_K_S")
client = Client(
    provider=provider,
    system_message=SystemMessage("You are a helpful coding assistant."),
)

response = client.call(ClientRequest(UserMessage("Write a python quicksort function.")))
print(response.get_message())
```

### 2. Continuing After Tool Execution

When tool calls are received in `response.get_tool_calls()`, tools are executed and results added as `ToolMessage` instances to client history. You can then invoke `continue_after_tools`:

```python
# Execute tool and append ToolMessage to client history
tool_result_msg = ToolMessage(
    content="Current temp is 22°C",
    tool_call_id=tool_call["id"],
    tool_name="weather",
)
client.get_history().append(tool_result_msg)

# Resume completion with updated history
final_response = client.continue_after_tools()
print(final_response.get_message())
```

---

## Conversation History Management

The internal [`MessageHistory`](file:///c:/Users/danie/Desktop/qs/qs/src/qs/core/models/history.py) enforces key conversation invariants:
- **First Message Rule**: The first message MUST be a `SystemMessage`.
- **System Message Immutability**: The initial system message cannot be replaced or deleted during execution.
- **Copying & Truncation**:
  - `client.get_history()`: Returns the live list of messages.
  - `client.copy_history()`: Returns a deep copy of current history.
  - `client.replace_history_from(start_index, messages)`: Replaces history starting from `start_index` while preserving earlier messages.

---

## Streaming Architecture

When streaming is enabled in `ClientRequest` or provider calls, events are pushed via a callback receiving [`ProviderStreamEvent`](file:///c:/Users/danie/Desktop/qs/qs/src/qs/core/models/provider_stream_event.py).

### Stream Event Types ([`ProviderStreamEventType`](file:///c:/Users/danie/Desktop/qs/qs/src/qs/core/models/provider_stream_event.py))
- `START`: Emitted when streaming begins.
- `REASONING`: Emitted token-by-token during model reasoning/thinking phases.
- `RESPONSE`: Emitted token-by-token during regular text content generation.
- `TOOL_CALL`: Emitted when tool call definitions are generated.
- `END`: Emitted when streaming completes normally.
- `ERROR`: Emitted if an error occurs during streaming.

```python
def on_stream_event(event: ProviderStreamEvent) -> None:
    if event.type == ProviderStreamEventType.REASONING:
        print(f"[Thinking]: {event.content}", end="", flush=True)
    elif event.type == ProviderStreamEventType.RESPONSE:
        print(event.content, end="", flush=True)


response = client.call_stream(
    ClientRequest(UserMessage("Explain general relativity")),
    on_event=on_stream_event,
)
```
