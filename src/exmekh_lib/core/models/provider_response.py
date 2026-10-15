from exmekh_lib.config.model_text import MODEL_TEXT
from exmekh_lib.core.models.assistant_message import AssistantMessage
from exmekh_lib.core.models.interactive import InterruptedMessage
from exmekh_lib.core.models.model_thinking import ModelThinking
from exmekh_lib.core.models.model_usage import ModelUsage
from exmekh_lib.core.models.tool_message import ToolMessage
from exmekh_lib.interfaces.message import Message
from exmekh_lib.utils.xml import stringify_tool_result


class ProviderResponse(AssistantMessage):
    def __init__(
        self,
        response: AssistantMessage,
        thinking: ModelThinking | None = None,
        usage: ModelUsage | None = None,
        interruptions: tuple[InterruptedMessage, ...] = (),
        superseded_response: AssistantMessage | None = None,
    ) -> None:
        if not isinstance(response, AssistantMessage):
            raise TypeError(MODEL_TEXT.response_must_be_an_assistantmessage)
        if superseded_response is not None and not isinstance(
            superseded_response, AssistantMessage
        ):
            raise TypeError(MODEL_TEXT.response_must_be_an_assistantmessage)

        content = response.get_message()
        if usage is not None and not isinstance(usage, ModelUsage):
            raise TypeError(MODEL_TEXT.usage_must_be_a_modelusage_instance)

        if thinking is not None and not isinstance(thinking, ModelThinking):
            raise TypeError(MODEL_TEXT.thinking_must_be_a_modelthinking_instance)

        super().__init__(
            content=content,
            name=response.get_name(),
            tool_calls=response.get_tool_calls(),
        )
        self.thinking = thinking
        self.usage = usage
        self.interruptions = interruptions
        # Completed server output remains transcript evidence after steering,
        # but its calls must never reach the execution dispatcher.
        self.superseded_response = superseded_response

    def history_messages(self) -> list[Message[object]]:
        """Expose transcript evidence independently of executable response calls."""
        messages: list[Message[object]] = []
        if self.superseded_response is not None:
            messages.append(self.superseded_response)
            for call in self.superseded_response.get_tool_calls() or ():
                messages.append(
                    ToolMessage(
                        stringify_tool_result(
                            {
                                "status": "cancelled",
                                "executed": False,
                                "code": "steered",
                                "message": MODEL_TEXT.tool_call_cancelled_by_steering,
                            }
                        ),
                        call.get_id(),
                        call.get_name(),
                    )
                )
        if (
            self.get_message() is not None
            or self.get_tool_calls()
            or not self.interruptions
        ):
            messages.append(self)
        return messages
