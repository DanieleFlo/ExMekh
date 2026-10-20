# Client & Messaging Model

The `exmekh_lib` framework provides a strongly-typed, object-oriented layer over raw LLM calls. The primary interaction point is the [`Client`](../src/exmekh_lib/core/models/client.py) class, which maintains conversation history, manages system prompts, and handles tool responses and event streaming.

---

## Messaging Hierarchy & Data Models

All chat messages in `exmekh_lib` implement the [`Message`](../src/exmekh_lib/interfaces/message.py) interface.

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

1. **[`SystemMessage`](../src/exmekh_lib/core/models/system_message.py)**: Sets agent role or instructions. Must be the first message in conversation history.
2. **[`UserMessage`](../src/exmekh_lib/core/models/user_message.py)**: Represents input from the user or prompt source. Supports plain text and structured content payloads.
3. **[`AssistantMessage`](../src/exmekh_lib/core/models/assistant_message.py)**: Output emitted by the LLM. Contains:
   - `content`: Text response string (if any).
   - `thinking`: Optional [`ModelThinking`](../src/exmekh_lib/core/models/model_thinking.py) containing reasoning/chain-of-thought tokens.
   - `tool_calls`: List of tool call specifications requested by the model.
4. **[`ToolMessage`](../src/exmekh_lib/core/models/tool_message.py)**: Contains the return output of a tool call, paired with `tool_call_id` and `tool_name`.
5. **[`FunctionMessage`](../src/exmekh_lib/core/models/function_message.py)**: Legacy function result message.
6. **[`StructuredAssistantMessage`](../src/exmekh_lib/core/models/structured_assistant_message.py)**: Wraps model responses when validated against a Pydantic schema via `call_structured`.

`ProviderResponse` and `StructuredProviderResponse` expose optional
[`ModelUsage`](../src/exmekh_lib/core/models/model_usage.py). Counts must be
nonnegative integers or `None`; booleans are rejected. `get_total_tokens()` uses
the reported total, including zero, or derives it when both input and output
counts are known. Otherwise it returns `None`.

An unfinished Responses completion raises `ProviderResponseIncompleteError`
(a `RuntimeError` subclass), so callers can classify it without inspecting error
text. Terminal streaming usage remains available before this error propagates.

---

## Client API Usage

### 1. Basic Synchronous Calls

```python
from exmekh_lib.core.models.client import Client
from exmekh_lib.core.models.client_request import ClientRequest
from exmekh_lib.core.models.system_message import SystemMessage
from exmekh_lib.core.models.user_message import UserMessage
from exmekh_lib.providers.local import LocalProvider

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

from exmekh_lib.core.models.client_request import ClientRequest
from exmekh_lib.core.models.user_message import (
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
    FilesUserMessage(
        [
            DocumentUserMessage(
                Path("notes.txt").read_bytes(), "notes", "txt", "text/plain"
            ),
        ]
    ),
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
`ExMekh-Lib` preserves that protocol and does not silently convert documents to text.
Document filename indices advance across the whole message, including separate
`FilesUserMessage` containers. Empty containers contribute no content parts or
extra blank lines to history dumps; explicit empty strings remain text parts.

### Rejected inputs and history recovery

DS4 accepts text and image content. Native `FilesUserMessage` attachments are
rejected with HTTP 400 and an explicit message: only text and images are
supported. Supply textual file contents through `TextsUserMessage`, or open
images with PIL and use `ImagesUserMessage`. `ExMekh-Lib` preserves native files for
other endpoints that support them; it does not extract documents automatically.

`LocalProvider` exposes HTTP 400 as
`exmekh_lib.interfaces.provider.ProviderRequestRejectedError`, a `RuntimeError` subclass
that preserves the server's error message and chains the original API error.
`Client.call`, `call_structured` and `continue_with_user_messages` remove only
messages added by the rejected call. Previous context and completed tool results
remain available, so the next request can proceed without manual history repair.
Streaming and persistent DeepAgent runs use the same recovery behavior.
Other provider errors retain unanswered input for retry as before. Existing
history supplied to a tool continuation is preserved on rejection.

```python
from exmekh_lib.interfaces.provider import ProviderRequestRejectedError

try:
    client.call(ClientRequest(UserMessage(FilesUserMessage([document]))))
except ProviderRequestRejectedError as error:
    print(str(error))

# The rejected document is no longer in this client's history.
response = client.call(
    ClientRequest(UserMessage("Continue with the previous context."))
)
```

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

The internal [`MessageHistory`](../src/exmekh_lib/core/models/history.py) enforces key conversation invariants:
- **First Message Rule**: The first message MUST be a `SystemMessage`.
- **Leading System Message**: History updates must retain a message with the system role at the beginning.
- **Copying & Truncation**:
  - `client.get_history()`: Returns a new list containing the existing message objects; use client methods to change history.
  - `client.copy_history()`: Returns a deep copy of current history.
  - `client.replace_history_from(start_index, messages)`: Replaces history starting from `start_index` while preserving earlier messages.

---

## Streaming Architecture

When streaming is enabled in `ClientRequest` or provider calls, a `ProviderEventHandler` instance receives each `ProviderStreamEvent` through its `handle` method.

Provider decorators can use `ProviderRequest.with_system_message(text)`,
`with_extra(mapping_or_none)`, `with_streaming(enabled)` and
`with_event_handler(handler_or_none)` to copy a request without expanding its
multimodal attachments twice. The copies preserve interactive handlers and
structured-response contracts; replacing the system text requires a leading
`SystemMessage`.

### Stream Event Types ([`ProviderStreamEventType`](../src/exmekh_lib/core/models/provider_stream_event.py))
- `START`: Emitted when streaming begins.
- `REASONING`: Emitted token-by-token during model reasoning/thinking phases.
- `RESPONSE`: Emitted token-by-token during regular text content generation.
- `TOTALS`: Provider token usage, including failed or incomplete Responses terminals
  when the provider supplies it. Observers can retain this even if parsing raises.
- `END`: End-of-stream notification; provider errors still propagate to the caller.
- `INTERACTIVE`: Structured interactive-protocol control metadata.
- `STEERING_ADDED`: A steering user message committed to the current history.

```python
from exmekh_lib.core.models.provider_stream_event import (
    ProviderStreamEvent,
    ProviderStreamEventType,
)
from exmekh_lib.interfaces.event_handler import ProviderEventHandler


class PrintProviderEvents(ProviderEventHandler):
    def handle(self, event: ProviderStreamEvent) -> None:
        if event.type == ProviderStreamEventType.REASONING:
            print(f"[Thinking]: {event.content}", end="", flush=True)
        elif event.type == ProviderStreamEventType.RESPONSE:
            print(event.content, end="", flush=True)


response = client.call(
    ClientRequest(
        UserMessage("Explain general relativity"),
        streaming=True,
        on_event=PrintProviderEvents(),
    )
)
```

### Reasoning override for one request

The optional `ClientRequest(..., reasoning_effort_override="none")` disables
reasoning for that request. Both `Client.call` and `Client.call_structured`
forward it without changing the provider configuration or subsequent requests.
The `read_tool_output` summarizer uses it on a dedicated client with the same
model as the node.
