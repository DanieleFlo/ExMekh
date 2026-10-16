"""Application decisions at the node's safe inference boundaries."""

from typing import Protocol

from exmekh_lib.core.models.provider_request import ProviderRequest


class HistoryCompactionPolicy(Protocol):
    def before_inference(self, request: ProviderRequest) -> bool:
        """Observe an eligible request and decide whether to attempt compaction."""
        ...

    def after_compaction(self, request: ProviderRequest) -> None:
        """Observe the applied context without requesting another compaction."""
        ...
