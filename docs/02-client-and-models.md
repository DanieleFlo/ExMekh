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

### Mixed user content

`UserMessage` accepts a single `str`, `TextsUserMessage`, `ImagesUserMessage`,
or `FilesUserMessage`, or one list containing any combination of those types.
`NodeUserMessage` uses the same content type, so `DeepAgent.run()` and
`steer()` accept that list directly.

The whole list becomes **one user message**. `UserMessage.iter_content_parts()`
expands its containers into ordered text, images and documents. Both provider
mappers and history dumps use that public iterator; they do not interpret the
containers independently. A list of strings becomes ordered text parts. A single string keeps
its existing plain-string representation. Nested lists and unsupported elements
are rejected before inference.

`ImagesUserMessage` contains only Pillow images. Text belongs in its own list
element; image and document loading belong to the caller.

```python
from pathlib import Path

from PIL import Image

from qs.core.models.client_request import ClientRequest
from qs.core.models.user_message import (
    DocumentUserMessage,
    FilesUserMessage,
    ImagesUserMessage,
    TextsUserMessage,
    UserMessage,
)

with Image.open("screenshot.png") as source:
    image = source.copy()

content = [
    "Inspect this screenshot.",
    ImagesUserMessage([image]),
    TextsUserMessage(["Compare it with the attached notes.", "Keep this order."]),
    FilesUserMessage([
        DocumentUserMessage(Path("notes.txt").read_bytes(), "notes", "txt", "text/plain"),
    ]),
    "Explain any differences.",
]
response = client.call(ClientRequest(UserMessage(content)))
# With a DeepAgent:
response = agent.run(content)
```

The same ordering applies to streaming and structured-response requests.
Steering copies the whole list, images, text containers, and documents before
queuing it. Guidance delivered inside a skill remains in that skill's archived
history after compaction, following the ordinary steering policy. History
objects retain the content; history exports display text and image/document
hashes in order and do not contain the binary data for restoring attachments.

Images are encoded as PNG data URLs and documents retain their native file
parts. The endpoint and model determine which attachment types they can read.
QS preserves that protocol and does not silently convert documents to text.
Document filename indices advance across the whole message, including separate
`FilesUserMessage` containers. Empty containers contribute no content parts or
extra blank lines to history dumps; explicit empty strings remain text parts.

### 2. Continuing After Tool Execution

When tool calls are received in `response.get_tool_calls()`, tools are executed and results added as `ToolMessage` instances to client history. You can then invoke `continue_after_tools`:

```python
# Execute tool and append ToolMessage to client history
tool_result_msg = ToolMessage(
    content="Current temp is 22°C",
    tool_call_id=tool_call.get_id(),
    name="weather",
)
client.add_tool_response(tool_result_msg)

# Resume completion with updated history
final_response = client.continue_after_tools()
print(final_response.get_message())
```

---

## Conversation History Management

The internal [`MessageHistory`](file:///c:/Users/danie/Desktop/qs/qs/src/qs/core/models/history.py) enforces key conversation invariants:
- **First Message Rule**: The first message MUST be a `SystemMessage`.
- **Leading System Message**: History updates must retain a message with the system role at the beginning.
- **Copying & Truncation**:
  - `client.get_history()`: Returns a new list containing the existing message objects; use client methods to change history.
  - `client.copy_history()`: Returns a deep copy of current history.
  - `client.replace_history_from(start_index, messages)`: Replaces history starting from `start_index` while preserving earlier messages.

---

## Streaming Architecture

When streaming is enabled in `ClientRequest` or provider calls, a `ProviderEventHandler` instance receives each `ProviderStreamEvent` through its `handle` method.

### Stream Event Types ([`ProviderStreamEventType`](file:///c:/Users/danie/Desktop/qs/qs/src/qs/core/models/provider_stream_event.py))
- `START`: Emitted when streaming begins.
- `REASONING`: Emitted token-by-token during model reasoning/thinking phases.
- `RESPONSE`: Emitted token-by-token during regular text content generation.
- `TOOL_CALL`: Emitted when tool call definitions are generated.
- `END`: Emitted when streaming completes normally.
- `ERROR`: Emitted if an error occurs during streaming.

```python
from qs.core.models.provider_stream_event import ProviderStreamEvent, ProviderStreamEventType
from qs.interfaces.event_handler import ProviderEventHandler


class PrintProviderEvents(ProviderEventHandler):
    def handle(self, event: ProviderStreamEvent) -> None:
        if event.type == ProviderStreamEventType.REASONING:
            print(f"[Thinking]: {event.content}", end="", flush=True)
        elif event.type == ProviderStreamEventType.RESPONSE:
            print(event.content, end="", flush=True)


response = client.call(ClientRequest(
    UserMessage("Explain general relativity"),
    streaming=True,
    on_event=PrintProviderEvents(),
))
```
