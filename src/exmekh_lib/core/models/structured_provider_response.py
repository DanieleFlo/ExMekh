from pydantic import BaseModel

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
            raise TypeError("Response must be a StructuredAssistantMessage")

        if usage is not None and not isinstance(usage, ModelUsage):
            raise TypeError("Usage must be a ModelUsage instance")

        content = response.get_message()
        if not isinstance(content, BaseModel):
            raise TypeError("Response content must be a Pydantic BaseModel")

        if thinking is not None and not isinstance(thinking, ModelThinking):
            raise TypeError("Thinking must be a ModelThinking instance")

        super().__init__(
            content=content,
            name=response.get_name(),
            tool_calls=response.get_tool_calls(),
        )
        self.thinking = thinking
        self.usage = usage
