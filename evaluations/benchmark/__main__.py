"""Standalone analytical/agent evaluation; pytest never starts model campaigns."""

import argparse
import importlib
import math
from contextlib import ExitStack
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import TypeAdapter

from evaluations.benchmark.campaign import (
    default_components,
    run_campaign,
)
from evaluations.benchmark.models import CampaignSettings, Suite, Variant
from exmekh_lib.config.provider import ProviderSettings
from exmekh_lib.core.models.reasoning_effort import ReasoningEffortLMH
from exmekh_lib.providers.local import LocalProvider
from exmekh_lib.utils.xml import stringify_tool_result

ROOT = Path(__file__).resolve().parents[1]


class EvaluationProvider(LocalProvider):
    def __init__(self, model: str, timeout: float, reasoning: str = "low") -> None:
        self.timeout = timeout
        super().__init__(
            model=model, api="responses", reasoning_effort=ReasoningEffortLMH(reasoning)
        )

    def _build_client(self) -> OpenAI:
        return super()._build_client().with_options(timeout=self.timeout, max_retries=0)


def main() -> int:
    load_dotenv(ROOT.parent / ".env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=ROOT / "suites/agent_tasks.json")
    parser.add_argument(
        "--profiles", type=Path, default=ROOT / "suites/handoff_variants.json"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/agent-benchmark.xml"
    )
    parser.add_argument("--model", default=ProviderSettings.from_environment().model)
    parser.add_argument("--judge-model")
    parser.add_argument(
        "--reasoning", choices=("none", "low", "medium", "high"), default="low"
    )
    parser.add_argument(
        "--judge-reasoning", choices=("none", "low", "medium", "high"), default="low"
    )
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--variant", action="append", default=[])
    parser.add_argument(
        "--split", choices=("development", "holdout", "all"), default="development"
    )
    parser.add_argument(
        "--plugin",
        action="append",
        default=[],
        help="Trusted Python module exposing register(components)",
    )
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument(
        "--token-penalty",
        type=float,
        default=1,
        help="Utility penalty per 1000 actor+judge tokens",
    )
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if (
        args.repetitions < 1
        or not math.isfinite(args.timeout)
        or args.timeout <= 0
        or not math.isfinite(args.token_penalty)
        or args.token_penalty < 0
    ):
        parser.error(
            "Use positive repetitions/timeout and a finite nonnegative token penalty"
        )
    suite = Suite.model_validate_json(args.suite.read_text(encoding="utf-8"))
    profiles = TypeAdapter(tuple[Variant, ...]).validate_json(
        args.profiles.read_text(encoding="utf-8")
    )
    available = {
        task.id
        for task in suite.tasks
        if args.split == "all" or task.split == args.split
    }
    if set(args.case) - available or set(args.variant) - {
        variant.id for variant in profiles
    }:
        parser.error("Unknown task/profile or task outside the selected split")
    suite = suite.model_copy(
        update={
            "tasks": tuple(
                task
                for task in suite.tasks
                if task.id in available and (not args.case or task.id in args.case)
            )
        }
    )
    profiles = tuple(
        variant
        for variant in profiles
        if not args.variant or variant.id in args.variant
    )
    if not suite.tasks or not profiles:
        parser.error("Selection is empty")
    components = default_components()
    for module in args.plugin:
        importlib.import_module(module).register(components)
    for task in suite.tasks:
        components.drivers.validate(task)
        components.checkers.validate(task)
    if args.list:
        print(
            stringify_tool_result(
                {
                    "suite": suite.model_dump(mode="json"),
                    "profiles": [item.model_dump(mode="json") for item in profiles],
                }
            )
        )
        return 0
    settings = CampaignSettings(
        args.output,
        args.repetitions,
        args.seed,
        args.token_penalty,
        actor_model=args.model,
        judge_model=args.judge_model or args.model,
        actor_reasoning=args.reasoning,
        judge_reasoning=args.judge_reasoning,
        provider_timeout_seconds=args.timeout,
        plugins=tuple(args.plugin),
    )
    with ExitStack() as stack:
        actor = stack.enter_context(
            EvaluationProvider(args.model, args.timeout, args.reasoning)
        )
        judge = stack.enter_context(
            EvaluationProvider(
                args.judge_model or args.model, args.timeout, args.judge_reasoning
            )
        )
        trials = run_campaign(suite, profiles, settings, actor, judge, components)
    return 2 if any(trial.infrastructure_error for trial in trials) else 0


if __name__ == "__main__":
    raise SystemExit(main())
