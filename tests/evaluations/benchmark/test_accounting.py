"""Accounting observes every provider call independently of mutable history."""

import pytest
from pydantic import BaseModel

from evaluations.benchmark.accounting import (
    MeteredProvider,
    RunBudgetExceeded,
)
from evaluations.benchmark.models import Limits, Variant
from exmekh_lib.core.models.assistant_message import (
    AssistantMessage,
    FunctionCall,
    ToolCall,
)
from exmekh_lib.core.models.model_usage import ModelUsage
from exmekh_lib.core.models.provider_request import ProviderRequest
from exmekh_lib.core.models.provider_response import ProviderResponse
from exmekh_lib.core.models.provider_stream_event import (
    ProviderStreamEvent,
    ProviderStreamEventType,
)
from exmekh_lib.core.models.structured_assistant_message import (
    StructuredAssistantMessage,
)
from exmekh_lib.core.models.structured_provider_request import StructuredProviderRequest
from exmekh_lib.core.models.structured_provider_response import (
    StructuredProviderResponse,
)
from exmekh_lib.core.models.system_message import SystemMessage
from exmekh_lib.interfaces.provider import Provider
from tests.tokenizer_support import StubTokenizer


class ScriptedProvider(Provider):
    def __init__(self, responses=(), usage=ModelUsage(10, 5, 15)):
        super().__init__(StubTokenizer())
        self.responses = iter(responses)
        self.requests = []
        self.usage = usage

    def get_tools_text(self, tools):
        return ""

    def call(self, request):
        self.requests.append(request)
        response = next(self.responses, AssistantMessage("Done"))
        if self.usage is not None:
            request.execute_on_stream_event(
                ProviderStreamEvent(ProviderStreamEventType.TOTALS, usage=self.usage)
            )
        if isinstance(response, BaseException):
            raise response
        return ProviderResponse(response, usage=self.usage)

    def call_structured(self, request):
        self.requests.append(request)
        return StructuredProviderResponse(
            StructuredAssistantMessage(request.get_response_model()(text="summary")),
            usage=self.usage,
        )


def call(name, arguments):
    return AssistantMessage(
        "", tool_calls=[ToolCall("call-" + name, FunctionCall(name, arguments))]
    )


def request():
    return ProviderRequest([SystemMessage("instructions")])


def test_generation_compaction_and_structured_calls_all_count():
    class Summary(BaseModel):
        text: str

    meter = MeteredProvider(
        ScriptedProvider(),
        Limits(compaction="allow"),
        Variant(id="candidate", system_addendum="Extra"),
    )
    original = request()
    estimated = meter.estimate_input_tokens(original)
    meter.call(original)
    assert meter.get_inferences()[0].estimated_input_tokens == estimated
    meter.begin_compaction()
    meter.call_structured(
        StructuredProviderRequest([SystemMessage("summarize")], Summary)
    )
    totals = meter.get_usage_totals()
    assert totals["total_tokens"] == 30
    assert totals["compaction_calls"] == 1
    assert meter.get_inferences()[1].request["messages"][0]["content"] == "summarize"
    assert original.get_messages()[0].get_message() == "instructions"
    assert original.get_extra() is None
    assert not original.get_streaming()


def test_failed_response_still_accounts_for_terminal_usage():
    meter = MeteredProvider(ScriptedProvider([RuntimeError("incomplete")]), Limits())
    with pytest.raises(RuntimeError):
        meter.call(request())
    assert meter.get_usage_totals()["total_tokens"] == 15
    assert meter.get_inferences()[0].error == "RuntimeError: incomplete"


def test_interruption_is_recorded_before_it_propagates():
    meter = MeteredProvider(ScriptedProvider([KeyboardInterrupt()]), Limits())
    with pytest.raises(KeyboardInterrupt):
        meter.call(request())
    assert meter.get_usage_totals()["total_tokens"] == 15
    assert meter.get_inferences()[0].error.startswith("KeyboardInterrupt")


def test_missing_usage_is_unknown_not_free():
    meter = MeteredProvider(ScriptedProvider(usage=None), Limits())
    meter.call(request())
    assert meter.get_usage_totals()["total_tokens"] is None
    assert meter.get_usage_totals()["unknown_usage_calls"] == 1
    assert meter.get_usage_totals()["estimated_input_tokens"] > 0


def test_call_budget_and_compaction_permission_are_enforced_before_inference():
    provider = ScriptedProvider()
    meter = MeteredProvider(provider, Limits(max_calls=1, compaction="forbid"))
    meter.call(request())
    with pytest.raises(RunBudgetExceeded, match="call budget"):
        meter.call(request())
    assert len(provider.requests) == 1
    meter.begin_compaction()
    with pytest.raises(RunBudgetExceeded, match="forbidden"):
        meter.call(request())


def test_response_over_budget_remains_in_the_cost_ledger():
    meter = MeteredProvider(
        ScriptedProvider(usage=ModelUsage(500, 500, 1000)), Limits(max_total_tokens=500)
    )
    with pytest.raises(RunBudgetExceeded, match="last response"):
        meter.call(request())
    assert meter.get_usage_totals()["total_tokens"] == 1000


def test_zero_reported_usage_is_never_replaced_by_an_estimate():
    meter = MeteredProvider(
        ScriptedProvider(usage=ModelUsage(0, 0, 0)), Limits(max_total_tokens=500)
    )
    for _ in range(10):
        meter.call(request())
    assert meter.get_usage_totals()["total_tokens"] == 0


def test_budget_reserves_the_declared_output_tokens_before_sending():
    provider = ScriptedProvider()
    meter = MeteredProvider(
        provider, Limits(max_context_tokens=1000, max_output_tokens=800)
    )
    with pytest.raises(RunBudgetExceeded, match="Context budget"):
        meter.call(ProviderRequest([SystemMessage("x" * 200)]))
    assert provider.requests == []
    assert meter.call_count == 0


def test_diagnostics_and_callbacks_cannot_mutate_the_cost_ledger():
    def modify(record):
        record.request["messages"].clear()
        record.response["tool_calls"].clear()

    meter = MeteredProvider(
        ScriptedProvider([call("calculate", {"value": 42})]), Limits(), on_record=modify
    )
    meter.call(request())
    records = meter.get_inferences()
    assert records[0].request["messages"]
    assert meter.get_tool_calls()[0]["arguments"] == {"value": 42}
    records[0].request["messages"].clear()
    calls = meter.get_tool_calls()
    calls[0]["arguments"]["value"] = 0
    assert meter.get_inferences()[0].request["messages"]
    assert meter.get_tool_calls()[0]["arguments"]["value"] == 42


def test_elapsed_budget_starts_at_first_inference(monkeypatch):
    elapsed = [0]
    monkeypatch.setattr(
        "evaluations.benchmark.accounting.monotonic", lambda: elapsed[0]
    )
    meter = MeteredProvider(ScriptedProvider(), Limits(max_seconds=10))
    elapsed[0] = 100
    meter.call(request())
    elapsed[0] = 111
    with pytest.raises(RunBudgetExceeded, match="Elapsed-time"):
        meter.call(request())


def test_wrapper_preserves_underlying_provider_identity_methods(monkeypatch):
    provider = ScriptedProvider()
    session = object()
    monkeypatch.setattr(provider, "generate_id", lambda: "provider-owned-id")
    monkeypatch.setattr(provider, "get_session", lambda: session)
    meter = MeteredProvider(provider, Limits())
    assert meter.generate_id() == "provider-owned-id"
    assert meter.get_session() is session
