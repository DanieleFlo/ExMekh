from qs.core.agents.deep_skill.execution import DeepSkillExecution
from qs.core.models.assistant_message import AssistantMessage
from qs.core.models.model_thinking import ModelThinking
from qs.core.models.model_usage import ModelUsage
from qs.core.models.provider_response import ProviderResponse


class AgentResponse(ProviderResponse):
    def __init__(
        self,
        response: AssistantMessage,
        history: list[dict[str, str]],
        skill_executions: tuple[DeepSkillExecution, ...] = (),
        usage: ModelUsage | None = None,
        thinking: ModelThinking | None = None,
    ) -> None:
        super().__init__(response, thinking=thinking, usage=usage)
        self.history = history
        self.skill_executions = skill_executions

    def get_history_tree(self) -> list[dict[str, str]]:
        return [execution.history_tree() for execution in self.skill_executions]
