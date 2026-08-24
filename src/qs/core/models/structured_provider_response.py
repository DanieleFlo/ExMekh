from pydantic import BaseModel

from qs.core.models.model_thinking import ModelThinking
from qs.core.models.model_usage import ModelUsage
from qs.core.models.structured_assistant_message import StructuredAssistantMessage


class StructuredProviderResponse[ResponseContent: BaseModel]:
    def __init__(
        self,
        response: StructuredAssistantMessage[ResponseContent],
        thinking: ModelThinking | None = None,
        usage: ModelUsage | None = None,
    ) -> None:
        self.response = response
        self.thinking = thinking
        self.usage = usage
        self._check_inputs()

    def _check_inputs(self) -> None:
        if not isinstance(self.response, StructuredAssistantMessage):
            raise TypeError("Response must be a StructuredAssistantMessage")

        if self.usage is not None and not isinstance(self.usage, ModelUsage):
            raise TypeError("Usage must be a ModelUsage instance")

        if not issubclass(type(self.response.get_message()), BaseModel):
            raise TypeError("Response content must be a Pydantic BaseModel")

        if self.thinking is not None and not isinstance(self.thinking, ModelThinking):
            raise TypeError("Thinking must be a ModelThinking instance")
