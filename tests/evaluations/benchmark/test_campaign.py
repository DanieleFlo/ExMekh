"""Extensibility, failure coverage and persisted costs need no model service."""

from dataclasses import replace
from xml.etree import ElementTree as ET

import pytest

from evaluations.benchmark.campaign import (
    Components,
    run_campaign,
)
from evaluations.benchmark.models import (
    Assessment,
    CampaignSettings,
    Check,
    Criterion,
    Observation,
    Suite,
    Task,
    Trial,
    Variant,
)
from evaluations.benchmark.reporting import summary
from tests.evaluations.benchmark.test_accounting import ScriptedProvider


def definitions():
    return Suite(
        name="arbitrary",
        tasks=(
            Task(
                id="custom",
                driver="custom",
                domain="any",
                prompt="A custom outcome",
                criteria=(
                    Criterion(
                        id="outcome",
                        description="A custom requirement",
                        verification="exact",
                        verification_reason="Directly observable",
                        checker="custom",
                        required=True,
                    ),
                ),
            ),
        ),
    )


def test_custom_driver_checker_and_frozen_report(tmp_path):
    components = Components()
    components.drivers.register(
        "custom",
        lambda task, variant, directory, provider: Observation("completed", "ok"),
    )
    components.checkers.register(
        "custom", lambda *args: Check(100, {"observable": True})
    )
    settings = CampaignSettings(tmp_path / "report.xml")
    actor = ScriptedProvider()
    trials = run_campaign(
        definitions(),
        (Variant(id="first"), Variant(id="second")),
        settings,
        actor,
        actor,
        components,
    )
    assert len(trials) == 2
    assert all(trial.assessment.success for trial in trials)
    assert ET.parse(settings.output).getroot().findtext("complete") == "true"
    with pytest.raises(FileExistsError):
        run_campaign(
            definitions(), (Variant(id="first"),), settings, actor, actor, components
        )


def test_judge_failure_preserves_analytic_evidence(tmp_path):
    class FailingJudge:
        def evaluate(self, *args):
            raise RuntimeError("judge unavailable")

    components = Components(judge=FailingJudge())
    components.drivers.register("custom", lambda *args: Observation("completed", "ok"))
    components.checkers.register("custom", lambda *args: Check(100, {"runs": True}))
    suite = definitions()
    item = suite.tasks[0].criteria[0].model_copy(update={"verification": "partial"})
    suite = suite.model_copy(
        update={"tasks": (suite.tasks[0].model_copy(update={"criteria": (item,)}),)}
    )
    actor = ScriptedProvider()
    trial = run_campaign(
        suite,
        (Variant(id="first"),),
        CampaignSettings(tmp_path / "report.xml"),
        actor,
        actor,
        components,
    )[0]
    assert trial.assessment.quality is None
    assert trial.assessment.criteria[0]["analytic"] == 100
    assert trial.infrastructure_error == "RuntimeError: judge unavailable"


def test_total_cost_includes_failed_attempts_and_reports_unknown_usage(tmp_path):
    usage = {"total_tokens": 1000, "known_total_tokens": 1000, "compaction_calls": 1}
    good = Trial(
        "task",
        "domain",
        "development",
        "first",
        1,
        Observation("completed"),
        Assessment(100, True, ()),
        usage,
        usage,
        1,
    )
    bad = replace(
        good,
        repetition=2,
        assessment=Assessment(None, None, ()),
        infrastructure_error="failure",
    )
    result = summary([good, bad], CampaignSettings(tmp_path / "report.xml"))[
        "variants"
    ]["first"]
    assert result["total_tokens"] == 4000
    assert result["tokens_per_success"] == 4000
    assert result["infrastructure_errors"] == 1
    missing = replace(bad, actor_usage={**usage, "total_tokens": None})
    result = summary([good, missing], CampaignSettings(tmp_path / "other.xml"))[
        "variants"
    ]["first"]
    assert result["total_tokens"] is None
    assert result["tokens_per_success"] is None
    assert result["complete_usage_trials"] == 1
    assert result["complete_actor_usage_trials"] == 1
    assert result["complete_judge_usage_trials"] == 2
    assert result["mean_judge_tokens"] == 1000


def test_success_count_does_not_require_optional_quality_evidence(tmp_path):
    usage = {"total_tokens": 0, "known_total_tokens": 0, "compaction_calls": 0}
    trial = Trial(
        "task",
        "domain",
        "development",
        "first",
        1,
        Observation("completed"),
        Assessment(None, True, ()),
        usage,
        usage,
        0,
    )
    result = summary([trial], CampaignSettings(tmp_path / "report.xml"))["variants"][
        "first"
    ]
    assert result["scored"] == 0
    assert result["successes"] == 1


def test_exact_tasks_do_not_invoke_a_model_judge(tmp_path):
    class UnneededJudge:
        def evaluate(self, *args):
            raise AssertionError("Exact checks must not call the judge")

    components = Components(judge=UnneededJudge())
    components.drivers.register("custom", lambda *args: Observation("completed", "ok"))
    components.checkers.register("custom", lambda *args: Check(100))
    provider = ScriptedProvider()
    trial = run_campaign(
        definitions(),
        (Variant(id="first"),),
        CampaignSettings(tmp_path / "report.xml"),
        provider,
        provider,
        components,
    )[0]
    assert trial.infrastructure_error is None
    assert trial.assessment.success is True
