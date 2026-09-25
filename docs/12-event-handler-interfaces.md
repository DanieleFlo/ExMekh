# Event Handler Interfaces

Implement stream event handlers by subclassing the appropriate interface from
`qs.interfaces.event_handler`. The runtime checks the `handle` method signature
and annotations when a handler is supplied.

Use `ProviderEventHandler` for `ClientRequest.on_event` and
`ProviderRequest.on_event`. Use `DeepAgentEventHandler` for
`DeepAgent.on_stream` or `Node.initialize(on_stream=...)`. Each class must
implement `handle` with the event type shown below and return `None`.

```python
from qs.core.agents.deep_agent import DeepAgentEvent, DeepAgentEventHandler
from qs.core.models.provider_stream_event import ProviderStreamEvent
from qs.interfaces.event_handler import ProviderEventHandler


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
