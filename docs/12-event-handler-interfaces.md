# Event Handler Interfaces

Implement stream event handlers by subclassing the appropriate interface from
`exmekh_lib.interfaces.event_handler`. The runtime checks the `handle` method signature
and annotations when a handler is supplied.

Use `ProviderEventHandler` for `ClientRequest.on_event` and
`ProviderRequest.on_event`. Use `DeepAgentEventHandler` for
`DeepAgent.on_stream` or `Node.initialize(on_stream=...)`. Each class must
implement `handle` with the event type shown below and return `None`.

```python
from exmekh_lib.core.agents.deep_agent import DeepAgentEvent, DeepAgentEventHandler
from exmekh_lib.core.models.provider_stream_event import ProviderStreamEvent
from exmekh_lib.interfaces.event_handler import ProviderEventHandler


class PrintAgentEvents(DeepAgentEventHandler):
    def handle(self, event: DeepAgentEvent) -> None:
        print(event)


class PrintProviderEvents(ProviderEventHandler):
    def handle(self, event: ProviderStreamEvent) -> None:
        print(event)
```

Pass instances, such as `DeepAgent(..., on_stream=PrintAgentEvents())` or
`ClientRequest(..., on_event=PrintProviderEvents())`. Invalid method signatures
or event annotations fail when the handler is supplied.

Raw functions are rejected at these registration points. Instantiate the
appropriate event handler class before passing it to the runtime.

`ProviderStreamEventType.STEERING_ADDED` (`"steering_added"`) is delivered to
the same `ProviderEventHandler` supplied through `on_event`. The client emits
one event per steering message after the batch is added to conversation history
and before calling the provider, including when streaming is disabled. The
`message` field contains an independent copy of the `UserMessage`, preserving
text, images, and documents. Other event types do not carry this field.
All messages in the committed batch are copied before invoking its first event
handler, so a callback cannot alter or drop later notifications by mutating the
source batch.

Queuing a message with `steer()` does not emit this event; pending input discarded
after a failure does not emit it either. Ordinary input passed to `run()` does
not emit it. For agents, the existing `DeepAgentStreamEvent` wraps this provider
event with the active `depth` and `skill_path`. Steering delivered inside a skill
is not emitted again when its transcript is compacted.
