from exmekh_lib.core.models.assistant_message import AssistantMessage
from exmekh_lib.core.models.provider_request import ProviderRequest
from exmekh_lib.core.models.tool_message import ToolMessage


def validate_request_history(request: ProviderRequest) -> None:
    """Reject histories that a model provider cannot continue from."""
    messages = request.get_messages()
    if messages[0].get_role_name() != "system":
        raise ValueError("The system prompt must be first")
    pending: dict[str, str] = {}

    for index, message in enumerate(messages):
        role = message.get_role_name()
        if index:
            previous_role = messages[index - 1].get_role_name()
            if previous_role == "system" and role != "user":
                raise ValueError("The system prompt must be followed by user input")

        if isinstance(message, ToolMessage):
            call_id = message.get_tool_call_id()
            if call_id not in pending:
                raise ValueError(
                    f"Orphan or duplicate tool result at {index}: {call_id}"
                )
            if pending.pop(call_id) != message.get_name():
                raise ValueError(f"Tool result name does not match call {call_id}")
            continue

        if pending:
            raise ValueError(
                f"{role} at {index} interrupts unanswered calls: {pending}"
            )
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"Unexpected message role at {index}: {role}")
        if role == "system" and index != 0:
            raise ValueError("The system prompt must be first")
        if isinstance(message, AssistantMessage):
            calls = message.get_tool_calls() or ()
            pending = {call.id: call.get_name() for call in calls}
            if len(pending) != len(calls):
                raise ValueError("Duplicate IDs within an assistant tool batch")

    if pending:
        raise ValueError(f"Unanswered tool calls at end of history: {pending}")

    scope = (request.get_extra() or {}).get("agentic")
    if not isinstance(scope, dict) or scope.get("operation") != "return":
        return
    skill_call_id = scope.get("skill_call_id")
    trailing_results = []
    for message in reversed(messages):
        if not isinstance(message, ToolMessage):
            break
        trailing_results.append(message.get_tool_call_id())
    if skill_call_id not in trailing_results:
        raise ValueError(
            f"Agentic return must end with a tool result for {skill_call_id}"
        )
