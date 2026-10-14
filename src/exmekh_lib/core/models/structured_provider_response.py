from pydantic import BaseModel

from exmekh_lib.config.model_text import MODEL_TEXT
from exmekh_lib.core.models.model_thinking import ModelThinking
from exmekh_lib.core.models.model_usage import ModelUsage
from exmekh_lib.core.models.structured_assistant_message import (
    StructuredAssistantMessage,
)


class StructuredProviderResponse[ResponseContent: BaseModel](
    StructuredAssistantMessage[ResponseContent]
):
    def __init__(
        self,
        response: StructuredAssistantMessage[ResponseContent],
        thinking: ModelThinking | None = None,
        usage: ModelUsage | None = None,
    ) -> None:
        if not isinstance(response, StructuredAssistantMessage):
            raise TypeError(MODEL_TEXT.response_must_be_a_structuredassistantmessage)

        if usage is not None and not isinstance(usage, ModelUsage):
            raise TypeError(MODEL_TEXT.usage_must_be_a_modelusage_instance)

        content = response.get_message()
        if not isinstance(content, BaseModel):
            raise TypeError(MODEL_TEXT.response_content_must_be_a_pydantic_basemodel)

        if thinking is not None and not isinstance(thinking, ModelThinking):
            raise TypeError(MODEL_TEXT.thinking_must_be_a_modelthinking_instance)

        super().__init__(
            content=content,
            name=response.get_name(),
            tool_calls=response.get_tool_calls(),
        )
        self.thinking = thinking
        self.usage = usage
