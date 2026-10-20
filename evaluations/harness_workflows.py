"""Score workflow decisions under prompt and parameter-documentation variants."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any, Literal
from xml.etree import ElementTree as ET

from dotenv import load_dotenv
from openai import OpenAIError

from evaluations.tool_argument_grading import RawToolCall
from evaluations.workflow_cases import (
    Step,
    WorkflowCase,
    WorkflowGrade,
    build_cases,
    grade_workflow,
)
from evaluations.workflow_prompts import (
    PromptBundle,
    capture_prompts,
    load_bundle,
    save_bundle,
)
from exmekh_lib.config.provider import ProviderSettings
from exmekh_lib.core.models.assistant_message import (
    AssistantMessage,
    FunctionCall,
    ToolCall,
)
from exmekh_lib.core.models.provider_request import ProviderRequest
from exmekh_lib.core.models.system_message import SystemMessage
from exmekh_lib.core.models.tool_message import ToolMessage
from exmekh_lib.core.models.user_message import UserMessage
from exmekh_lib.core.providers.to_openai_response_request_mapper import (
    ToOpenAIResponseRequestMapper,
)
from exmekh_lib.interfaces.message import Message
from exmekh_lib.providers.local import LocalProvider
from exmekh_lib.utils.json import serialize_json
from exmekh_lib.utils.xml import append_value_element, render_xml

Variant = Literal["baseline", "workflow", "compact", "schema"]
VARIANTS: tuple[Variant, ...] = ("baseline", "workflow", "compact")
ALL_VARIANTS: tuple[Variant, ...] = (*VARIANTS, "schema")
REPOSITORY = Path(__file__).resolve().parents[1]
BASELINE = Path(__file__).parent / "fixtures" / "harness-prompts-before.xml"


@dataclass(frozen=True)
class WorkflowSettings:
    model: str
    output: Path
    cases: tuple[str, ...]
    repetitions: int = 1
    seed: int = 20261020
    variants: tuple[Variant, ...] = VARIANTS
    reasoning: str = "low"
    timeout: float = 90
    max_output_tokens: int = 1024
    candidate: Path | None = None


@dataclass(frozen=True)
class Trial:
    case: str
    category: str
    variant: Variant
    repetition: int
    grade: WorkflowGrade | None
    elapsed_seconds: float
    prompt_tokens: int
    infrastructure_error: str | None = None
    domain: str = "harness"
    split: str = "development"


def variant_bundles(
    current: PromptBundle, schema: PromptBundle | None = None
) -> dict[Variant, PromptBundle]:
    baseline = load_bundle(BASELINE)
    bundles: dict[Variant, PromptBundle] = {
        "baseline": baseline,
        "workflow": PromptBundle(current.system, baseline.tools, baseline.skills),
        "compact": current,
    }
    if schema is not None:
        bundles["schema"] = schema
    return bundles


def build_payload(
    case: WorkflowCase, bundle: PromptBundle, settings: WorkflowSettings
) -> dict[str, Any]:
    messages: list[Message[object]] = [
        SystemMessage(bundle.system),
        UserMessage(case.task),
    ]
    for index, step in enumerate(case.steps):
        call_id = f"step-{index}"
        messages.append(
            AssistantMessage(
                "",
                tool_calls=[ToolCall(call_id, FunctionCall(step.name, step.arguments))],
            )
        )
        messages.append(
            ToolMessage(
                step.output if step.output is not None else bundle.skills[step.name],
                call_id,
                step.name,
            )
        )
    request = ProviderRequest(messages)
    payload = ToOpenAIResponseRequestMapper.map_request(
        request, model=settings.model, reasoning_effort=settings.reasoning
    )
    # Scope masks and required tool choice would conceal routing/termination errors.
    payload["tools"] = [
        *bundle.tools,
        {
            "type": "function",
            "name": "review-stage",
            "description": "Open the application's review scope, which exposes file-processing.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            "strict": True,
        },
    ]
    payload["tool_choice"] = "auto"
    payload["max_output_tokens"] = settings.max_output_tokens
    return payload


def execute_trial(
    provider: LocalProvider,
    settings: WorkflowSettings,
    case: WorkflowCase,
    bundle: PromptBundle,
    variant: Variant,
    repetition: int,
) -> tuple[Trial, dict[str, Any]]:
    payload = build_payload(case, bundle, settings)
    prompt_tokens = provider.get_tokenizer().count_tokens(
        serialize_json({"input": payload["input"], "tools": payload["tools"]})
    )
    trace: dict[str, Any] = {"request": payload, "case": asdict(case)}
    started = perf_counter()
    error = None
    grade = None
    try:
        response = provider.client.with_options(
            timeout=settings.timeout, max_retries=0
        ).responses.create(**payload)
    except OpenAIError as failure:
        error = f"{type(failure).__name__}: {failure}"
    else:
        trace["response"] = response.model_dump(mode="json")
        grade = grade_response(case, trace["response"])
    trial = Trial(
        case.name,
        case.category,
        variant,
        repetition,
        grade,
        perf_counter() - started,
        prompt_tokens,
        error,
        case.domain,
        case.split,
    )
    trace["trial"] = asdict(trial)
    return trial, trace


def grade_response(case: WorkflowCase, response: dict[str, Any]) -> WorkflowGrade:
    if response["status"] != "completed":
        return WorkflowGrade(0, {"completed_response": False}, response["status"])
    calls = tuple(
        RawToolCall(item["name"], item["arguments"])
        for item in response["output"]
        if item["type"] == "function_call"
    )
    text = "".join(
        content["text"]
        for item in response["output"]
        if item["type"] == "message"
        for content in item["content"]
        if content["type"] == "output_text"
    )
    return grade_workflow(case, calls, text)


def summarize(trials: list[Trial]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for variant in ALL_VARIANTS:
        samples = [trial for trial in trials if trial.variant == variant]
        scored = [trial for trial in samples if trial.grade is not None]
        result[variant] = {
            "attempts": len(samples),
            "scored": len(scored),
            "infrastructure_errors": sum(
                trial.infrastructure_error is not None for trial in samples
            ),
            "mean_score": fmean(
                trial.grade.score for trial in scored if trial.grade is not None
            )
            if scored
            else None,
            "fully_correct": sum(
                trial.grade is not None and all(trial.grade.checks.values())
                for trial in scored
            ),
            "mean_prompt_tokens": fmean(trial.prompt_tokens for trial in samples)
            if samples
            else None,
        }
    pairs: dict[tuple[str, int], dict[Variant, float]] = {}
    for trial in trials:
        if trial.grade is not None:
            pairs.setdefault((trial.case, trial.repetition), {})[trial.variant] = (
                trial.grade.score
            )
    result["paired_comparison"] = {}
    for variant in ("workflow", "compact"):
        deltas = [
            pair[variant] - pair["baseline"]
            for pair in pairs.values()
            if variant in pair and "baseline" in pair
        ]
        result["paired_comparison"][variant] = {
            "pairs": len(deltas),
            "mean_delta_from_baseline": fmean(deltas) if deltas else None,
        }
    location_deltas = [
        pair["schema"] - pair["compact"]
        for pair in pairs.values()
        if "schema" in pair and "compact" in pair
    ]
    result["parameter_location_comparison"] = {
        "pairs": len(location_deltas),
        "mean_schema_minus_skill": fmean(location_deltas) if location_deltas else None,
    }
    result["by_split"] = {}
    for split in ("development", "holdout"):
        result["by_split"][split] = {}
        for variant in ALL_VARIANTS:
            samples = [
                trial.grade
                for trial in trials
                if trial.split == split
                and trial.variant == variant
                and trial.grade is not None
            ]
            result["by_split"][split][variant] = {
                "scored": len(samples),
                "mean_score": fmean(grade.score for grade in samples)
                if samples
                else None,
                "fully_correct": sum(all(grade.checks.values()) for grade in samples),
            }
    return result


def write_report(
    settings: WorkflowSettings,
    trials: list[Trial],
    *,
    complete: bool,
    infrastructure_error: str | None = None,
    source_trace: Path | None = None,
) -> None:
    root = ET.Element(
        "harness-workflow-evaluation", {"version": "1", "grading-version": "2"}
    )
    append_value_element(
        root, "source_trace", str(source_trace) if source_trace else None
    )
    append_value_element(root, "settings", asdict(settings))
    append_value_element(
        root,
        "cases",
        [asdict(case) for case in build_cases() if case.name in settings.cases],
    )
    append_value_element(root, "complete", complete)
    append_value_element(root, "infrastructure_error", infrastructure_error)
    append_value_element(
        root,
        "planned_samples",
        len(settings.cases) * len(settings.variants) * settings.repetitions,
    )
    append_value_element(root, "summary", summarize(trials))
    append_value_element(root, "trials", [asdict(trial) for trial in trials])
    temporary = settings.output.with_suffix(".xml.tmp")
    temporary.write_text(render_xml(root) + "\n", encoding="utf-8")
    temporary.replace(settings.output)


def run(settings: WorkflowSettings) -> list[Trial]:
    if settings.output.suffix != ".xml":
        raise ValueError("Report must have an .xml suffix")
    trace_path = settings.output.with_suffix(".jsonl")
    candidate_path = settings.output.with_suffix(".prompts.xml")
    schema_path = settings.output.with_suffix(".schema.prompts.xml")
    if any(
        path.exists()
        for path in (settings.output, trace_path, candidate_path, schema_path)
    ):
        raise FileExistsError("Choose unused report, trace and prompt snapshot paths")
    settings.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="workflow-prompts-") as directory:
        current = (
            load_bundle(settings.candidate)
            if settings.candidate
            else capture_prompts(Path(directory))
        )
        schema = None
        if "schema" in settings.variants:
            production = (
                capture_prompts(Path(directory)) if settings.candidate else current
            )
            if current != production:
                raise ValueError(
                    "Schema comparison requires the frozen candidate to match current production prompts"
                )
            schema = capture_prompts(
                Path(directory), parameter_descriptions_in_schema=True
            )
    bundles = variant_bundles(current, schema)
    save_bundle(current, candidate_path)
    if schema is not None:
        save_bundle(schema, schema_path)
    trials: list[Trial] = []
    with (
        trace_path.open("x", encoding="utf-8") as traces,
        LocalProvider(model=settings.model, api="responses") as provider,
    ):
        write_report(settings, trials, complete=False)
        for repetition in range(1, settings.repetitions + 1):
            cases = [case for case in build_cases() if case.name in settings.cases]
            random.Random(settings.seed + repetition).shuffle(cases)
            for index, case in enumerate(cases):
                variants = list(settings.variants)
                offset = (index + repetition - 1) % len(variants)
                for variant in variants[offset:] + variants[:offset]:
                    trial, trace = execute_trial(
                        provider, settings, case, bundles[variant], variant, repetition
                    )
                    trials.append(trial)
                    traces.write(serialize_json(trace) + "\n")
                    traces.flush()
                    write_report(settings, trials, complete=False)
                    progress = ET.Element("workflow-evaluation-progress")
                    append_value_element(progress, "trial", asdict(trial))
                    print(render_xml(progress), flush=True)
    write_report(settings, trials, complete=True)
    return trials


def regrade(settings: WorkflowSettings, trace_path: Path) -> list[Trial]:
    """Review every saved response with current criteria, without inference."""
    if settings.output.suffix != ".xml" or settings.output.exists():
        raise ValueError("Choose an unused .xml report path")
    records = [
        json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()
    ]
    if not records:
        raise ValueError("Trace contains no trials")
    trials = []
    for record in records:
        case_data = record["case"]
        case = WorkflowCase(
            **(
                case_data
                | {"steps": tuple(Step(**step) for step in case_data["steps"])}
            )
        )
        grade = (
            grade_response(case, record["response"]) if "response" in record else None
        )
        trials.append(Trial(**(record["trial"] | {"grade": grade})))
    # Retain only facts recoverable from the trace; scheduling and timeout settings
    # are supplied by the caller and do not affect offline grading.
    settings = WorkflowSettings(
        model=records[0]["request"]["model"],
        output=settings.output,
        cases=tuple(dict.fromkeys(trial.case for trial in trials)),
        repetitions=max(trial.repetition for trial in trials),
        seed=settings.seed,
        variants=tuple(
            variant
            for variant in ALL_VARIANTS
            if any(trial.variant == variant for trial in trials)
        ),
        reasoning=records[0]["request"]
        .get("reasoning", {})
        .get("effort", settings.reasoning),
        timeout=settings.timeout,
        max_output_tokens=records[0]["request"]["max_output_tokens"],
        candidate=settings.candidate,
    )
    settings.output.parent.mkdir(parents=True, exist_ok=True)
    expected = len(settings.cases) * len(settings.variants) * settings.repetitions
    unique = {(trial.case, trial.variant, trial.repetition) for trial in trials}
    write_report(
        settings,
        trials,
        complete=len(unique) == len(trials) == expected,
        source_trace=trace_path,
    )
    return trials


def main() -> int:
    load_dotenv(REPOSITORY / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument(
        "--regrade", type=Path, help="Regrade a saved JSONL trace without inference"
    )
    parser.add_argument("--model", default=ProviderSettings.from_environment().model)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY / "evaluations/results/harness-workflows.xml",
    )
    parser.add_argument("--case", action="append", default=[], dest="cases")
    parser.add_argument(
        "--variant", action="append", choices=ALL_VARIANTS, dest="variants"
    )
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument(
        "--reasoning", choices=("none", "low", "medium", "high"), default="low"
    )
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument(
        "--candidate", type=Path, help="Use a frozen candidate prompt snapshot"
    )
    parser.add_argument(
        "--split", choices=("all", "development", "holdout"), default="all"
    )
    args = parser.parse_args()
    names = tuple(
        case.name
        for case in build_cases()
        if args.split == "all" or case.split == args.split
    )
    if args.list:
        root = ET.Element("workflow-cases")
        append_value_element(
            root,
            "cases",
            [asdict(case) for case in build_cases() if case.name in names],
        )
        print(render_xml(root))
        return 0
    if (
        set(args.cases) - set(names)
        or args.repetitions < 1
        or not 0 < args.timeout < float("inf")
        or args.max_output_tokens < 1
    ):
        parser.error(
            "Use known cases, positive repetitions, finite positive timeout and positive output limit"
        )
    settings = WorkflowSettings(
        args.model,
        args.output,
        tuple(args.cases) or names,
        args.repetitions,
        args.seed,
        tuple(dict.fromkeys(args.variants or VARIANTS)),
        args.reasoning,
        args.timeout,
        args.max_output_tokens,
        args.candidate,
    )
    try:
        trials = regrade(settings, args.regrade) if args.regrade else run(settings)
    except OpenAIError as error:
        diagnostic = f"{type(error).__name__}: {error}"
        write_report(settings, [], complete=False, infrastructure_error=diagnostic)
        failure = ET.Element("workflow-evaluation-failed")
        append_value_element(failure, "infrastructure_error", diagnostic)
        print(render_xml(failure))
        return 2
    return 2 if any(trial.infrastructure_error for trial in trials) else 0


if __name__ == "__main__":
    raise SystemExit(main())
