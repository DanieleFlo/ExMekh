"""Apply the library compactor before benchmark histories exhaust their context."""

from evaluations.benchmark.accounting import MeteredProvider
from exmekh_lib.core.agents.node import Node
from exmekh_lib.core.models.provider_request import ProviderRequest
from exmekh_lib.core.models.request_budget import RequestBudget


class ContextCompaction:
    def __init__(self, provider: MeteredProvider) -> None:
        self.provider = provider
        self.scheduled_done = False

    def before_inference(self, request: ProviderRequest) -> bool:
        limits = self.provider.limits
        scheduled = (
            not self.scheduled_done
            and limits.compact_after_calls is not None
            and self.provider.call_count >= limits.compact_after_calls
        )
        near_capacity = (
            self.provider.estimate_input_tokens(request)
            >= limits.max_context_tokens * limits.compaction_threshold
        )
        return scheduled or near_capacity

    def after_compaction(self, request: ProviderRequest) -> None:
        self.scheduled_done = True


def configure_compaction(node: Node, provider: MeteredProvider) -> None:
    if provider.limits.compaction == "forbid":
        return
    node.set_history_compaction_policy(
        ContextCompaction(provider),
        budget=RequestBudget(
            provider.limits.max_context_tokens,
            provider.estimate_input_tokens,
            response_reserve=provider.limits.max_output_tokens,
        ),
    )
    for child in node.children:
        configure_compaction(child, provider)
