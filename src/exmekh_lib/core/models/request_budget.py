"""A caller-supplied context capacity and estimator for complete requests."""

from collections.abc import Callable
from dataclasses import dataclass, field

from exmekh_lib.config.history_compaction import HISTORY_COMPACTION
from exmekh_lib.core.models.provider_request import ProviderRequest
from exmekh_lib.interfaces.tokenizer import Tokenizer


class ContextBudgetExceededError(RuntimeError):
    """No bounded request can be prepared without discarding protected input."""


@dataclass(frozen=True, slots=True)
class RequestBudget:
    capacity: int
    count_tokens: Callable[[ProviderRequest], int]
    response_reserve: int | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if type(self.capacity) is not int or self.capacity <= 0:
            raise ValueError(HISTORY_COMPACTION.invalid_budget)
        if self.response_reserve is not None and (
            type(self.response_reserve) is not int
            or not 0 <= self.response_reserve < self.capacity
        ):
            raise ValueError(HISTORY_COMPACTION.invalid_response_reserve)

    @property
    def prompt_limit(self) -> int:
        # Keep generation room even when a tool batch jumps across the threshold.
        reserve = self.response_reserve
        if reserve is None:
            reserve = min(
                HISTORY_COMPACTION.response_reserve_tokens, self.capacity // 8
            )
        return self.capacity - reserve

    def fits(self, request: ProviderRequest) -> bool:
        return self.count_tokens(request) <= self.prompt_limit

    def require_fits(self, request: ProviderRequest) -> None:
        tokens = self.count_tokens(request)
        if tokens > self.prompt_limit:
            raise ContextBudgetExceededError(
                HISTORY_COMPACTION.budget_exceeded.format(
                    limit=self.prompt_limit, tokens=tokens
                )
            )

    def split_text(
        self,
        text: str,
        tokenizer: Tokenizer,
        make_request: Callable[[str], ProviderRequest],
    ) -> tuple[str, ...]:
        """Split losslessly, checking the actual request rather than additive costs."""
        self.require_fits(make_request(""))
        pending = [text]
        chunks: list[str] = []
        while pending:
            part = pending.pop()
            request = make_request(part)
            if self.fits(request):
                chunks.append(part)
                continue
            size = max(1, tokenizer.count_tokens(part) // 2)
            pieces = tokenizer.split_text(part, size)
            if len(pieces) < 2:
                self.require_fits(request)
            pending.extend(reversed(pieces))
        return tuple(chunks)
