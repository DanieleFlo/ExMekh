"""Fixed diagnostic and log messages for the interface domain."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class InterfaceText:
    after_must_be_an_aftertool_instance: str = "after must be an AfterTool instance"
    before_must_be_a_beforetool_instance: str = "before must be a BeforeTool instance"
    event_handler_annotations_could_not_be_resolved: str = (
        "Event handler annotations could not be resolved"
    )
    event_handler_handle_must_be_a_synchronous_method: str = (
        "Event handler handle must be a synchronous method"
    )
    event_handler_handle_must_be_callable: str = "Event handler handle must be callable"
    event_handler_must_accept_exactly_one_event: str = (
        "Event handler must accept exactly one event"
    )
    event_handler_must_be_a_instance: str = (
        "event handler must be a {interface_name} instance"
    )
    event_handler_must_define_handle_event_none: str = (
        "Event handler must define handle(event: {event_name}) -> None"
    )
    this_provider_does_not_implement_tools_request_serialization: str = (
        "This provider does not implement tools request serialization"
    )
    tokenizer_must_be_a_tokenizer_instance: str = (
        "Tokenizer must be a Tokenizer instance"
    )
    tool_hook_handle_must_accept_argument_s: str = (
        "Tool hook handle must accept {argument_count} argument(s)"
    )
    tool_hook_handle_must_be_synchronous: str = "Tool hook handle must be synchronous"


INTERFACE_TEXT = InterfaceText()
