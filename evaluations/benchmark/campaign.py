"""Run arbitrary suites with frozen profiles, isolated trials and paired reports."""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from time import monotonic
from typing import Protocol
from uuid import uuid4

from evaluations.benchmark.accounting import (
    Inference,
    MeteredProvider,
    RunBudgetExceeded,
)
from evaluations.benchmark.checks import (
    check_answer,
    check_calls,
    check_events,
    check_fluid,
    check_json,
    check_python,
    check_python_raises,
    check_unchanged,
)
from evaluations.benchmark.drivers import (
    DriverRegistry,
    checkpoint_score,
    run_checkpoint,
    run_native,
)
from evaluations.benchmark.judge import LibraryJudge
from evaluations.benchmark.models import (
    Assessment,
    CampaignSettings,
    Check,
    JudgeMark,
    Observation,
    Suite,
    Task,
    Trial,
    Variant,
)
from evaluations.benchmark.reporting import (
    code_hashes,
    render_progress,
    write_report,
    write_trial,
)
from evaluations.benchmark.scoring import CheckerRegistry, combine
from exmekh_lib.interfaces.provider import Provider
from exmekh_lib.utils.json import serialize_json


class Judge(Protocol):
    def evaluate(
        self,
        task: Task,
        observation: Observation,
        checks: dict[str, Check],
        directory: Path,
        provider: MeteredProvider,
    ) -> tuple[JudgeMark, ...]: ...


@dataclass
class Components:
    drivers: DriverRegistry = field(default_factory=DriverRegistry)
    checkers: CheckerRegistry = field(default_factory=CheckerRegistry)
    judge: Judge = field(default_factory=LibraryJudge)


def default_components() -> Components:
    components = Components()
    components.drivers.register("native", run_native)
    components.drivers.register("checkpoint", run_checkpoint)
    for name, checker in (
        ("answer", check_answer),
        ("json", check_json),
        ("calls", check_calls),
        ("events", check_events),
        ("python", check_python),
        ("python_raises", check_python_raises),
        ("fluid", check_fluid),
        ("workflow", checkpoint_score),
        ("unchanged", check_unchanged),
    ):
        components.checkers.register(name, checker)
    return components


def run_trial(
    task: Task,
    variant: Variant,
    repetition: int,
    directory: Path,
    actor: Provider,
    judge: Provider,
    components: Components,
    settings: CampaignSettings,
    record: Callable[[str, Inference], None],
) -> Trial:
    directory.mkdir(parents=True)
    metered = MeteredProvider(
        actor, task.limits, variant, lambda item: record("actor", item)
    )
    evaluator = MeteredProvider(
        judge, settings.judge_limits, on_record=lambda item: record("judge", item)
    )
    started = monotonic()
    observation = Observation("not_started")
    assessment = Assessment(None, None, ())
    error = None
    try:
        observation = components.drivers.run(task, variant, directory, metered)
        if observation.status != "completed":
            assessment = Assessment(0, False, (), observation.error)
        else:
            checks = components.checkers.evaluate(task, observation, directory / "work")
            assessment = combine(task, checks, ())
            if any(check.score is None for check in checks.values()):
                error = "Analytical verification is unavailable"
            elif any(criterion.analytic_weight < 1 for criterion in task.criteria):
                marks = components.judge.evaluate(
                    task, observation, checks, directory, evaluator
                )
                assessment = combine(task, checks, marks)
            if error is None:
                error = assessment.error
    except RunBudgetExceeded as failure:
        if observation.status == "not_started":
            observation = Observation("budget_exceeded", error=str(failure))
            assessment = Assessment(0, False, (), str(failure))
        else:
            error = "Judge budget exhausted: " + str(failure)
    except KeyboardInterrupt:
        error = "Campaign interrupted; the active call may have unreported usage"
        observation = replace(observation, status="interrupted", error=error)
    except Exception as failure:
        error = f"{type(failure).__name__}: {failure}"
    trial = Trial(
        task.id,
        task.domain,
        task.split,
        variant.id,
        repetition,
        observation,
        assessment,
        metered.get_usage_totals(),
        evaluator.get_usage_totals(),
        monotonic() - started,
        error,
        str(directory.resolve()),
    )
    write_trial(trial, directory / "trial.xml")
    return trial


def run_campaign(
    suite: Suite,
    variants: tuple[Variant, ...],
    settings: CampaignSettings,
    actor: Provider,
    judge: Provider,
    components: Components | None = None,
) -> list[Trial]:
    components = components or default_components()
    if not variants or len({variant.id for variant in variants}) != len(variants):
        raise ValueError("Supply distinct variants")
    if settings.repetitions < 1 or settings.output.suffix != ".xml":
        raise ValueError("Supply positive repetitions and an XML report path")
    if (
        not math.isfinite(settings.token_penalty_per_1000)
        or settings.token_penalty_per_1000 < 0
    ):
        raise ValueError("Token penalty must be finite and nonnegative")
    trace_path = settings.output.with_suffix(".jsonl")
    runs = settings.output.with_suffix(".runs")
    if any(path.exists() for path in (settings.output, trace_path, runs)):
        raise FileExistsError("Choose unused report, trace and trial paths")
    for task in suite.tasks:
        components.drivers.validate(task)
        components.checkers.validate(task)
    settings.output.parent.mkdir(parents=True, exist_ok=True)
    runs.mkdir()
    manifest = {
        "suite": suite.model_dump(mode="json"),
        "variants": [variant.model_dump(mode="json") for variant in variants],
        "settings": {
            **asdict(settings),
            "output": str(settings.output),
            "judge_limits": settings.judge_limits.model_dump(mode="json"),
        },
        "planned_trials": len(suite.tasks) * len(variants) * settings.repetitions,
        "code_sha256": code_hashes(),
    }
    trials: list[Trial] = []
    write_report(settings, manifest, trials, False)
    with trace_path.open("x", encoding="utf-8") as traces:
        for repetition in range(1, settings.repetitions + 1):
            tasks = list(suite.tasks)
            random.Random(settings.seed + repetition).shuffle(tasks)
            for position, task in enumerate(tasks):
                offset = (position + repetition - 1) % len(variants)
                for variant in variants[offset:] + variants[:offset]:

                    def record(stage: str, inference: Inference) -> None:
                        traces.write(
                            serialize_json(
                                {
                                    "task": task.id,
                                    "variant": variant.id,
                                    "repetition": repetition,
                                    "stage": stage,
                                    "inference": asdict(inference),
                                }
                            )
                            + "\n"
                        )
                        traces.flush()

                    directory = runs / f"{task.id}-{repetition}-{uuid4().hex}"
                    trial = run_trial(
                        task,
                        variant,
                        repetition,
                        directory,
                        actor,
                        judge,
                        components,
                        settings,
                        record,
                    )
                    trials.append(trial)
                    write_report(settings, manifest, trials, False)
                    print(render_progress(trial), flush=True)
                    if trial.observation.status == "interrupted":
                        return trials
    write_report(settings, manifest, trials, True)
    return trials
