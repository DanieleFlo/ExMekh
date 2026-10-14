from exmekh_lib.config.model_text import MODEL_TEXT
from exmekh_lib.core.models.assistant_message import AssistantMessage
from exmekh_lib.core.models.interactive import InterruptedMessage
from exmekh_lib.core.models.model_thinking import ModelThinking
from exmekh_lib.core.models.model_usage import ModelUsage


class ProviderResponse(AssistantMessage):
    def __init__(
        self,
        response: AssistantMessage,
        thinking: ModelThinking | None = None,
        usage: ModelUsage | None = None,
        interruptions: tuple[InterruptedMessage, ...] = (),
    ) -> None:
        if not isinstance(response, AssistantMessage):
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
