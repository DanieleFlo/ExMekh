"""Existing decision cases use the same extensible campaign driver interface."""

import json

import pytest

from evaluations.benchmark.accounting import MeteredProvider
from evaluations.benchmark.drivers import checkpoint_score, run_checkpoint
from evaluations.benchmark.models import Criterion, Limits, Task, Variant
from evaluations.workflow_cases import build_cases
from exmekh_lib.core.models.assistant_message import AssistantMessage
from tests.evaluations.benchmark.test_accounting import ScriptedProvider, call
from tests.evaluations.test_harness_workflows import gold_call


@pytest.mark.parametrize("case", build_cases(), ids=lambda case: case.name)
def test_existing_decision_case_is_available_without_native_execution(tmp_path, case):
    task = Task(
        id=case.name,
        domain="checkpoint",
        driver="checkpoint",
        prompt="Inspect notes.md before reading it.",
        parameters={"case": case.name},
        criteria=(
            Criterion(
                id="decision",
                description="Correct next action",
                verification="exact",
                verification_reason="Predefined decision checkpoint",
                checker="workflow",
            ),
        ),
    )
    expected = gold_call(case) if case.next_tools else None
    provider = ScriptedProvider(
        [
            call(expected.name, json.loads(expected.arguments))
            if expected
            else AssistantMessage(" ".join(case.contains))
        ]
    )
    observation = run_checkpoint(
        task,
        Variant(id="current"),
        tmp_path,
        MeteredProvider(provider, Limits(max_context_tokens=100_000)),
    )
    assert checkpoint_score(task, observation, tmp_path, {}).score == 100
    assert len(provider.requests) == 1
