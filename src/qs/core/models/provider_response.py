from qs.core.models.model_thinking import ModelThinking
from qs.core.models.model_usage import ModelUsage
from qs.interfaces.message import Message


class ProviderResponse:
    def __init__(
        self,
        response: Message,
        thinking: ModelThinking | None = None,
        usage: ModelUsage | None = None,
    ) -> None:
        self.response = response
        self.thinking = thinking
        self.usage = usage
