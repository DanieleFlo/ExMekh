"""Run paired argument-compilation evaluations independently of pytest/live tests."""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean, pstdev
from time import perf_counter
from typing import Any, Literal
from xml.etree import ElementTree as ET

from dotenv import load_dotenv
from openai import OpenAIError

from evaluations.tool_argument_cases import ArgumentCase, build_cases
from evaluations.tool_argument_grading import (
    ArgumentGrade,
    RawToolCall,
    grade_arguments,
)
from exmekh_lib.config.provider import ProviderSettings
from exmekh_lib.config.runtime import RUNTIME
from exmekh_lib.config.tokenizer import QWEN_TOKENIZER
from exmekh_lib.core.agents.capabilities.catalog_attacher import SkillCatalogAttacher
from exmekh_lib.core.agents.capabilities.data import CapabilitiesData
from exmekh_lib.core.agents.capabilities.index import CapabilitiesIndex
from exmekh_lib.core.agents.deep_skill.catalog import SkillCatalog, SkillCatalogEntry
from exmekh_lib.core.agents.deep_skill.deep_skill import DeepSkill
from exmekh_lib.core.agents.deep_skill.metadata import DeepSkillMetadata
from exmekh_lib.core.agents.hds.hds import HDS
from exmekh_lib.core.agents.hds.metadata import HDSMetadata
from exmekh_lib.core.models.provider_request import ProviderRequest
from exmekh_lib.core.models.system_message import SystemMessage
from exmekh_lib.core.models.tools import Tools
from exmekh_lib.core.models.user_message import UserMessage
from exmekh_lib.core.providers.to_openai_response_request_mapper import (
    ToOpenAIResponseRequestMapper,
)
from exmekh_lib.providers.local import LocalProvider
from exmekh_lib.utils.json import serialize_json
from exmekh_lib.utils.xml import append_value_element, render_xml

Variant = Literal["full", "scoped"]
REPOSITORY = Path(__file__).resolve().parents[1]
COMMON_INSTRUCTIONS = (
    "You are already inside the active skill. Compile exactly one call to the "
    "available tool using the user request and parameter documentation. Do not "
    "answer with prose. Unless explicitly requested otherwise, use the default "
    "execution options. The evaluation records arguments; no tool is executed."
)


@dataclass(frozen=True)
class EvaluationSettings:
    model: str
    repetitions: int
    seed: int
    cases: tuple[str, ...]
    reasoning: str
    timeout_seconds: float
    max_output_tokens: int
    output: Path


@dataclass(frozen=True)
class TrialResult:
    case: str
    category: str
    variant: Variant
    repetition: int
    expected: dict[str, Any]
    calls: tuple[RawToolCall, ...]
    grade: ArgumentGrade | None
    elapsed_seconds: float
    tool_definition_tokens: int
    instruction_tokens: int
    response_status: str
    usage: dict[str, Any] | None
    infrastructure_error: str | None = None


def build_request(case: ArgumentCase, variant: Variant) -> ProviderRequest:
    skill = DeepSkill(
        name="argument-evaluation",
        markdown_path=Path("argument-evaluation.md"),
        metadata=DeepSkillMetadata(
            description="Compile the requested tool arguments.",
            sub_skills=(),
            sub_tools=(case.tool.get_name(),),
            inherits_parent_tools=False,
        ),
        body="Use the documented tool to satisfy the request.",
        references=(),
    )
    data = CapabilitiesData(deep_skills=(skill,), tools=(case.tool,))
    hds = HDS(
        name="argument-evaluation-root",
        markdown_path=Path("argument-evaluation-root.md"),
        metadata=HDSMetadata(
            description="Evaluation root.",
            sub_skills=(skill.get_name(),),
            sub_tools=(),
            exit_tool=RUNTIME.exit_tool_name,
        ),
        body="Compile the requested arguments.",
        references=(),
    )
    contextual = next(
        tool
        for tool in data.get_contextual_tools(hds)
        if tool.get_name() == case.tool.get_name()
    )
    tool = case.tool.with_description(contextual.get_description())
    if variant == "scoped":
        tool = contextual
        SkillCatalogAttacher(CapabilitiesIndex.build(data)).attach_to_all(
            data.deep_skills
        )
    else:
        catalog = SkillCatalog(
            sub_tools=(
                SkillCatalogEntry(case.tool.get_name(), case.tool.get_description()),
            )
        )
        skill.attach_catalog(catalog)
    instructions = skill.render_instructions("")
    return ProviderRequest(
        messages=[
            SystemMessage(
                f"{COMMON_INSTRUCTIONS}\n\n{RUNTIME.harness_output_guidance}\n\n{instructions}"
            ),
            UserMessage(case.prompt),
        ],
        tools=Tools([tool]),
        tool_choice="required",
    )


def trial_schedule(
    settings: EvaluationSettings,
) -> list[tuple[int, ArgumentCase, Variant]]:
    schedule: list[tuple[int, ArgumentCase, Variant]] = []
    for repetition in range(settings.repetitions):
        cases = [
            case
            for case in build_cases(settings.seed + repetition)
            if case.name in settings.cases
        ]
        random.Random(settings.seed + repetition).shuffle(cases)
        for index, case in enumerate(cases):
            variants: tuple[Variant, Variant] = (
                ("full", "scoped")
                if (index + repetition) % 2 == 0
                else ("scoped", "full")
            )
            schedule.extend((repetition + 1, case, variant) for variant in variants)
    return schedule


def execute_trial(
    provider: LocalProvider,
    settings: EvaluationSettings,
    case: ArgumentCase,
    variant: Variant,
    repetition: int,
) -> tuple[TrialResult, dict[str, Any]]:
    request = build_request(case, variant)
    payload = ToOpenAIResponseRequestMapper.map_request(
        request, model=settings.model, reasoning_effort=settings.reasoning
    )
    payload["max_output_tokens"] = settings.max_output_tokens
    tokenizer = provider.get_tokenizer()
    tool_tokens = tokenizer.count_tokens(serialize_json(payload["tools"]))
    instruction_tokens = tokenizer.count_tokens(
        "\n".join(str(message.get_message()) for message in request.get_messages())
    )
    started = perf_counter()
    trace: dict[str, Any] = {"request": payload}
    calls: tuple[RawToolCall, ...] = ()
    grade: ArgumentGrade | None = None
    usage: dict[str, Any] | None = None
    status = "request_error"
    infrastructure_error: str | None = None
    try:
        # Keep raw SDK arguments: LocalProvider.call() intentionally coerces values
        # for execution and would conceal some compilation errors in this eval.
        response = provider.client.with_options(
            timeout=settings.timeout_seconds, max_retries=0
        ).responses.create(**payload)
    except OpenAIError as error:
        infrastructure_error = f"{type(error).__name__}: {error}"
    else:
        calls = tuple(
            RawToolCall(item.name, item.arguments)
            for item in response.output
            if item.type == "function_call"
        )
        grade = (
            grade_arguments(case, calls)
            if response.status == "completed"
            else ArgumentGrade(0.0, False, (), f"Response status: {response.status}")
        )
        status = response.status or "unknown"
        usage = (
            response.usage.model_dump(mode="json")
            if response.usage is not None
            else None
        )
        trace["response"] = response.model_dump(mode="json")
    result = TrialResult(
        case=case.name,
        category=case.category,
        variant=variant,
        repetition=repetition,
        expected=case.expected_arguments(),
        calls=calls,
        grade=grade,
        elapsed_seconds=perf_counter() - started,
        tool_definition_tokens=tool_tokens,
        instruction_tokens=instruction_tokens,
        response_status=status,
        usage=usage,
        infrastructure_error=infrastructure_error,
    )
    trace["result"] = asdict(result)
    return result, trace


def summarize(results: list[TrialResult]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for variant in ("full", "scoped"):
        trials = [result for result in results if result.variant == variant]
        grades = [trial.grade for trial in trials if trial.grade is not None]
        scores = [grade.score for grade in grades]
        summary[variant] = {
            "attempts": len(trials),
            "scored_samples": len(scores),
            "infrastructure_errors": sum(
                trial.infrastructure_error is not None for trial in trials
            ),
            "mean_score": fmean(scores) if scores else None,
            "score_stddev": pstdev(scores) if scores else None,
            "exact_call_rate_percent": 100.0
            * sum(grade.exact for grade in grades)
            / len(grades)
            if grades
            else None,
            "mean_tool_definition_tokens": fmean(
                trial.tool_definition_tokens for trial in trials
            )
            if trials
            else None,
            "mean_instruction_tokens": fmean(
                trial.instruction_tokens for trial in trials
            )
            if trials
            else None,
        }
    paired: dict[tuple[str, int], dict[str, float]] = {}
    for trial in results:
        if trial.grade is not None:
            paired.setdefault((trial.case, trial.repetition), {})[trial.variant] = (
                trial.grade.score
            )
    deltas = [
        pair["scoped"] - pair["full"]
        for pair in paired.values()
        if "full" in pair and "scoped" in pair
    ]
    summary["comparison"] = {
        "paired_samples": len(deltas),
        "mean_score_delta_scoped_minus_full": fmean(deltas) if deltas else None,
    }
    case_scores: dict[str, Any] = {}
    for name in sorted({trial.case for trial in results}):
        case_scores[name] = {}
        for variant in ("full", "scoped"):
            scores = [
                trial.grade.score
                for trial in results
                if trial.case == name
                and trial.variant == variant
                and trial.grade is not None
            ]
            case_scores[name][variant] = {
                "mean_score": fmean(scores) if scores else None,
                "scored_samples": len(scores),
            }
    summary["cases"] = case_scores
    return summary


def write_report(
    settings: EvaluationSettings, results: list[TrialResult], *, complete: bool
) -> None:
    root = ET.Element("tool-argument-evaluation", {"version": "1"})
    append_value_element(root, "settings", asdict(settings))
    append_value_element(root, "tokenizer", QWEN_TOKENIZER.repository)
    append_value_element(root, "api", "responses")
    append_value_element(
        root, "planned_samples", len(settings.cases) * settings.repetitions * 2
    )
    append_value_element(root, "complete", complete)
    append_value_element(root, "summary", summarize(results))
    append_value_element(root, "trials", [asdict(result) for result in results])
    temporary = settings.output.with_suffix(settings.output.suffix + ".tmp")
    temporary.write_text(render_xml(root) + "\n", encoding="utf-8")
    temporary.replace(settings.output)


def run_evaluation(settings: EvaluationSettings) -> list[TrialResult]:
    if settings.output.suffix.lower() != ".xml":
        raise ValueError("The XML report path must have an .xml extension.")
    settings.output.parent.mkdir(parents=True, exist_ok=True)
    trace_path = settings.output.with_suffix(".jsonl")
    if settings.output.exists() or trace_path.exists():
        raise FileExistsError(f"Choose a new report path: {settings.output}")
    results: list[TrialResult] = []
    with trace_path.open("x", encoding="utf-8") as traces:
        write_report(settings, results, complete=False)
        with LocalProvider(model=settings.model, api="responses") as provider:
            for repetition, case, variant in trial_schedule(settings):
                result, trace = execute_trial(
                    provider, settings, case, variant, repetition
                )
                results.append(result)
                traces.write(serialize_json(trace) + "\n")
                traces.flush()
                write_report(settings, results, complete=False)
                progress = ET.Element("evaluation-progress")
                append_value_element(
                    progress,
                    "sample",
                    {
                        "case": case.name,
                        "variant": variant,
                        "repetition": repetition,
                        "score": result.grade.score if result.grade else None,
                        "infrastructure_error": result.infrastructure_error,
                    },
                )
                print(render_xml(progress), flush=True)
    write_report(settings, results, complete=True)
    return results


def main() -> int:
    load_dotenv(REPOSITORY / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list", action="store_true", help="List cases without contacting a model."
    )
    parser.add_argument("--model", default=ProviderSettings.from_environment().model)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--case", action="append", default=[], dest="cases")
    parser.add_argument(
        "--reasoning", default="low", choices=["none", "low", "medium", "high"]
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    default_output = (
        REPOSITORY
        / "evaluations"
        / "results"
        / f"tool-arguments-{datetime.now(UTC):%Y%m%dT%H%M%S%fZ}.xml"
    )
    parser.add_argument("--output", type=Path, default=default_output)
    args = parser.parse_args()
    cases = build_cases(args.seed)
    if args.list:
        root = ET.Element("tool-argument-cases")
        append_value_element(
            root,
            "cases",
            [{"name": case.name, "category": case.category} for case in cases],
        )
        print(render_xml(root))
        return 0
    if (
        args.repetitions < 1
        or not math.isfinite(args.timeout)
        or args.timeout <= 0
        or args.max_output_tokens < 1
    ):
        parser.error("Repetitions, timeout and max-output-tokens must be positive.")
    selected = tuple(
        case.name for case in cases if not args.cases or case.name in args.cases
    )
    unknown = set(args.cases) - {case.name for case in cases}
    if unknown:
        parser.error(f"Unknown cases: {', '.join(sorted(unknown))}")
    settings = EvaluationSettings(
        model=args.model,
        repetitions=args.repetitions,
        seed=args.seed,
        cases=selected,
        reasoning=args.reasoning,
        timeout_seconds=args.timeout,
        max_output_tokens=args.max_output_tokens,
        output=args.output,
    )
    results = run_evaluation(settings)
    root = ET.Element("evaluation-summary")
    append_value_element(root, "scores", summarize(results))
    append_value_element(root, "report", str(settings.output))
    print(render_xml(root))
    return (
        2 if any(result.infrastructure_error is not None for result in results) else 0
    )


if __name__ == "__main__":
    raise SystemExit(main())
