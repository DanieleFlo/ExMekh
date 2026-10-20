"""Real library agents run against deterministic scripted providers, never a server."""

import json

import pytest
from pydantic import TypeAdapter

from evaluations.benchmark.accounting import MeteredProvider
from evaluations.benchmark.campaign import default_components
from evaluations.benchmark.drivers import run_native
from evaluations.benchmark.judge import LibraryJudge, VerdictInput, submit_verdict
from evaluations.benchmark.models import Criterion, Limits, Observation, Task, Variant
from exmekh_lib.core.models.assistant_message import AssistantMessage
from exmekh_lib.interfaces.provider import ProviderResponseIncompleteError
from tests.evaluations.benchmark.test_accounting import ScriptedProvider, call


def test_native_skill_lifecycle_produces_observed_calls(tmp_path):
    provider = ScriptedProvider(
        [
            call("deep-thinking", {}),
            call("calculate", {"operation": "evaluate", "expression": "19+23"}),
            call("exit_agent", {"text": "42", "items": []}),
            AssistantMessage("42"),
        ]
    )
    meter = MeteredProvider(provider, Limits(max_context_tokens=100_000))
    task = Task(
        id="sum",
        domain="math",
        prompt="Compute 19+23",
        criteria=(
            Criterion(
                id="answer",
                description="42",
                verification="exact",
                verification_reason="Arithmetic",
                checker="answer",
                parameters={"number": 42},
            ),
        ),
    )
    result = run_native(task, Variant(id="current"), tmp_path, meter)
    assert result.status == "completed"
    assert result.answer == "42"
    assert not any(event.get("failed") for event in result.events)
    assert meter.get_usage_totals()["total_tokens"] == 60
    assert [item["name"] for item in result.calls] == [
        "deep-thinking",
        "calculate",
        "exit_agent",
    ]
    assert (
        default_components()
        .checkers.evaluate(task, result, tmp_path / "work")["answer"]
        .score
        == 100
    )


@pytest.mark.parametrize("wrapped", [False, True])
def test_native_driver_classifies_incomplete_responses_by_type(tmp_path, wrapped):
    failure = ProviderResponseIncompleteError("truncated without diagnostic wording")
    if wrapped:
        wrapper = RuntimeError("compaction failed")
        wrapper.__cause__ = failure
        failure = wrapper
    provider = ScriptedProvider([failure])
    task = Task(
        id="truncated",
        domain="runtime",
        prompt="Answer briefly",
        criteria=(
            Criterion(
                id="answer",
                description="Answer",
                verification="exact",
                verification_reason="Observable",
                checker="answer",
            ),
        ),
    )
    result = run_native(
        task, Variant(id="current"), tmp_path, MeteredProvider(provider, Limits())
    )
    assert result.status == "incomplete_output"
    assert result.error == "truncated without diagnostic wording"


def test_library_judge_records_validated_blinded_evidence(tmp_path):
    provider = ScriptedProvider(
        [
            AssistantMessage(
                json.dumps(
                    {
                        "marks": [
                            {
                                "criterion": "meaning",
                                "score": 80,
                                "evidence": ["answer"],
                                "rationale": "Explains heat flow; reversal is incomplete.",
                            }
                        ]
                    }
                )
            ),
        ]
    )
    meter = MeteredProvider(provider, Limits(max_context_tokens=100_000))
    task = Task(
        id="explain",
        domain="physics",
        prompt="Explain heat transfer",
        files={"reference.txt": "Heat flows from warm to cold."},
        criteria=(
            Criterion(
                id="meaning",
                description="Physically correct",
                verification="semantic",
                verification_reason="Requires content assessment",
            ),
        ),
    )
    marks = LibraryJudge().evaluate(
        task, Observation("completed", "Heat flows"), {}, tmp_path, meter
    )
    assert marks[0].score == 80
    assert len(meter.get_inferences()) == 1
    assert "input:reference.txt" in str(
        provider.requests[0].get_messages()[-1].get_message()
    )
    assert "variant" not in (tmp_path / "judge/contract.json").read_text()


def test_judge_contract_rejects_forged_ids_duplicate_marks_and_rewrites(tmp_path):
    (tmp_path / "contract.json").write_text(
        json.dumps({"criteria": ["meaning"], "evidence": ["answer"]})
    )
    valid = VerdictInput("meaning", 80, ["answer"], "Concrete reason")
    for marks in (
        [VerdictInput("meaning", 100, ["invented"], "Claim")],
        [valid, valid],
    ):
        with pytest.raises(ValueError):
            submit_verdict(tmp_path, marks)
    submit_verdict(tmp_path, [valid])
    with pytest.raises(FileExistsError):
        submit_verdict(tmp_path, [valid])


@pytest.mark.parametrize("score", [-5, 101, float("nan"), True, "80", 80.5])
def test_judge_contract_rejects_invalid_numeric_scores(tmp_path, score):
    (tmp_path / "contract.json").write_text(
        json.dumps({"criteria": ["meaning"], "evidence": ["answer"]})
    )
    with pytest.raises(ValueError):
        submit_verdict(
            tmp_path, [VerdictInput("meaning", score, ["answer"], "Concrete reason")]
        )


def test_judge_rejects_unexpected_verdict_fields():
    with pytest.raises(ValueError):
        TypeAdapter(list[VerdictInput]).validate_python(
            [
                {
                    "criterion": "meaning",
                    "score": 80,
                    "evidence": ["answer"],
                    "rationale": "Concrete reason",
                    "invented": "ignored?",
                }
            ]
        )
