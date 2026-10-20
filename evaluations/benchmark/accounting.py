"""Append-only provider accounting survives skill exits and history compaction."""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from time import monotonic
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from evaluations.benchmark.models import Limits, Variant
from exmekh_lib.core.models.model_usage import ModelUsage
from exmekh_lib.core.models.provider_request import ProviderRequest
from exmekh_lib.core.models.provider_response import ProviderResponse
from exmekh_lib.core.models.provider_stream_event import (
    ProviderStreamEvent,
    ProviderStreamEventType,
)
from exmekh_lib.core.models.structured_provider_request import StructuredProviderRequest
from exmekh_lib.core.models.structured_provider_response import (
    StructuredProviderResponse,
)
from exmekh_lib.core.models.tools import Tools
from exmekh_lib.interfaces.event_handler import ProviderEventHandler
from exmekh_lib.interfaces.provider import Provider
from exmekh_lib.utils.json import serialize_json

if TYPE_CHECKING:
    from exmekh_lib.core.cache.session_cache import SessionCache


class RunBudgetExceeded(RuntimeError):
    """A candidate exhausted a predeclared benchmark budget."""


class UsageTap(ProviderEventHandler):
    def __init__(self, delegate: ProviderEventHandler | None) -> None:
        self.delegate = delegate
        self.usage: ModelUsage | None = None
        self.text: list[str] = []
        self.reasoning: list[str] = []

    def handle(self, event: ProviderStreamEvent) -> None:
        if event.type is ProviderStreamEventType.TOTALS:
            self.usage = event.usage
        elif event.type is ProviderStreamEventType.RESPONSE:
            self.text.append(event.content or "")
        elif event.type is ProviderStreamEventType.REASONING:
            self.reasoning.append(event.content or "")
        if self.delegate:
            self.delegate.handle(event)


@dataclass(frozen=True)
class Inference:
    index: int
    structured: bool
    purpose: str
    request: dict[str, Any]
    response: dict[str, Any] | None
    estimated_input_tokens: int
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    elapsed_seconds: float
    error: str | None
    streamed_text: str = ""
    streamed_reasoning: str = ""

    @property
    def budgeted_tokens(self) -> int:
        return (
            self.total_tokens
            if self.total_tokens is not None
            else self.estimated_input_tokens
        )


class MeteredProvider(Provider):
    def __init__(
        self,
        provider: Provider,
        limits: Limits,
        variant: Variant | None = None,
        on_record: Callable[[Inference], None] | None = None,
    ) -> None:
        super().__init__(provider.get_tokenizer())
        self._provider = provider
        self.limits = limits
        self._variant = variant
        self._records: list[Inference] = []
        self._started: float | None = None
        self._compaction_active = False
        self._on_record = on_record

    @property
    def call_count(self) -> int:
        return len(self._records)

    def begin_compaction(self) -> None:
        self._compaction_active = True

    def end_compaction(self) -> None:
        self._compaction_active = False

    def get_inferences(self) -> tuple[Inference, ...]:
        """Return detached diagnostics; callers cannot rewrite the cost ledger."""
        return deepcopy(tuple(self._records))

    def get_usage_totals(self) -> dict[str, Any]:
        def total(field: str) -> int | None:
            values = [getattr(item, field) for item in self._records]
            return sum(values) if all(value is not None for value in values) else None

        return {
            "calls": self.call_count,
            "input_tokens": total("input_tokens"),
            "output_tokens": total("output_tokens"),
            "total_tokens": total("total_tokens"),
            "known_total_tokens": sum(item.total_tokens or 0 for item in self._records),
            "unknown_usage_calls": sum(
                item.total_tokens is None for item in self._records
            ),
            "estimated_input_tokens": sum(
                item.estimated_input_tokens for item in self._records
            ),
            "compaction_calls": sum(
                item.purpose == "compaction" for item in self._records
            ),
        }

    def get_tool_calls(self) -> tuple[dict[str, Any], ...]:
        return tuple(
            {"inference": record.index, **deepcopy(call)}
            for record in self._records
            if record.response
            for call in record.response.get("tool_calls", ())
        )

    def _elapsed_seconds(self) -> float:
        return monotonic() - self._started if self._started is not None else 0

    def _consumed_tokens(self) -> int:
        return sum(item.budgeted_tokens for item in self._records)

    def get_tools_text(self, tools: Tools | None) -> str:
        return self._provider.get_tools_text(tools)

    def generate_id(self) -> str:
        return self._provider.generate_id()

    def get_session(self) -> SessionCache:
        return self._provider.get_session()

    def _prepare_request(self, request: ProviderRequest) -> ProviderRequest:
        request = request.with_extra(
            {
                **(request.get_extra() or {}),
                "max_output_tokens": self.limits.max_output_tokens,
            }
        )
        # Terminal streaming usage remains observable even when output parsing fails.
        request = request.with_streaming(True)
        messages = request.get_messages()
        if (
            self._variant
            and self._variant.system_addendum
            and not self._compaction_active
            and request.get_reasoning_effort_override() is None
        ):
            # Copying preserves interactive handlers and structured-response contracts.
            request = request.with_system_message(
                str(messages[0].get_message()) + "\n" + self._variant.system_addendum
            )
        return request

    def _request_data(self, request: ProviderRequest) -> dict[str, Any]:
        data: dict[str, Any] = {
            "messages": [item.message_dump() for item in request.get_messages()],
            "tools_json": self.get_tools_text(request.get_tools()),
            "extra": request.get_extra(),
            "tool_choice": request.get_tool_choice(),
            "stream": request.get_streaming(),
            "reasoning_effort_override": request.get_reasoning_effort_override(),
        }
        if isinstance(request, StructuredProviderRequest):
            data["response_schema"] = request.get_response_model().model_json_schema()
        return data

    def estimate_input_tokens(self, request: ProviderRequest) -> int:
        """Count the same prepared request used for inference, without spending budget."""
        data = self._request_data(self._prepare_request(request))
        return self.get_tokenizer().count_tokens(serialize_json(data))

    def _prepare(
        self, request: ProviderRequest
    ) -> tuple[ProviderRequest, dict[str, Any], int]:
        if self._started is None:
            self._started = monotonic()
        if self._compaction_active and self.limits.compaction == "forbid":
            raise RunBudgetExceeded("History compaction is forbidden for this trial")
        request = self._prepare_request(request)
        data = self._request_data(request)
        estimate = self.get_tokenizer().count_tokens(serialize_json(data))
        if self.call_count >= self.limits.max_calls:
            raise RunBudgetExceeded("Provider call budget exhausted")
        if self._elapsed_seconds() >= self.limits.max_seconds:
            raise RunBudgetExceeded("Elapsed-time budget exhausted")
        if estimate > self.limits.prompt_limit:
            raise RunBudgetExceeded("Context budget exceeded before inference")
        if self._consumed_tokens() + estimate > self.limits.max_total_tokens:
            raise RunBudgetExceeded("Total-token budget exhausted")
        return request, data, estimate

    def _record(
        self,
        data: dict[str, Any],
        estimate: int,
        started: float,
        response: ProviderResponse | StructuredProviderResponse[Any] | None,
        error: BaseException | None,
        structured: bool,
        tap: UsageTap,
    ) -> None:
        usage = response.usage if response and response.usage else tap.usage
        prompt = usage.prompt_tokens if usage else None
        output = usage.completion_tokens if usage else None
        total = usage.get_total_tokens() if usage else None
        content = response.get_message() if response else None
        self._records.append(
            Inference(
                index=self.call_count + 1,
                structured=structured,
                purpose="compaction" if self._compaction_active else "generation",
                request=data,
                response={
                    "content": content.model_dump()
                    if isinstance(content, BaseModel)
                    else content,
                    "tool_calls": [
                        {
                            "id": call.get_id(),
                            "name": call.get_name(),
                            "arguments": call.get_arguments(),
                        }
                        for call in response.get_tool_calls() or ()
                    ],
                }
                if response
                else None,
                estimated_input_tokens=estimate,
                input_tokens=prompt,
                output_tokens=output,
                total_tokens=total,
                elapsed_seconds=monotonic() - started,
                error=f"{type(error).__name__}: {error}" if error else None,
                streamed_text="".join(tap.text),
                streamed_reasoning="".join(tap.reasoning),
            )
        )
        if self._on_record:
            self._on_record(deepcopy(self._records[-1]))

    def call(self, request: ProviderRequest) -> ProviderResponse:
        request, data, estimate = self._prepare(request)
        tap = UsageTap(request.on_event)
        request = request.with_event_handler(tap)
        started = monotonic()
        try:
            response = self._provider.call(request)
        except BaseException as error:
            self._record(data, estimate, started, None, error, False, tap)
            raise
        self._record(data, estimate, started, response, None, False, tap)
        self._check_completion_budget()
        return response

    def call_structured[ResponseModel: BaseModel](
        self, request: StructuredProviderRequest[ResponseModel]
    ) -> StructuredProviderResponse[ResponseModel]:
        prepared, data, estimate = self._prepare(request)
        assert isinstance(prepared, StructuredProviderRequest)
        tap = UsageTap(prepared.on_event)
        prepared = prepared.with_event_handler(tap)
        started = monotonic()
        try:
            response = self._provider.call_structured(prepared)
        except BaseException as error:
            self._record(data, estimate, started, None, error, True, tap)
            raise
        self._record(data, estimate, started, response, None, True, tap)
        self._check_completion_budget()
        return response

    def _check_completion_budget(self) -> None:
        if self._consumed_tokens() > self.limits.max_total_tokens:
            raise RunBudgetExceeded("Total-token budget exceeded by the last response")
        if self._elapsed_seconds() > self.limits.max_seconds:
            raise RunBudgetExceeded("Elapsed-time budget exceeded by the last response")
