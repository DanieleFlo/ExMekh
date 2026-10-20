"""Campaign statistics and XML persistence, independent of agent execution."""

import hashlib
from dataclasses import asdict
from pathlib import Path
from statistics import fmean, stdev
from typing import Any
from xml.etree import ElementTree as ET

from evaluations.benchmark.models import CampaignSettings, Trial
from exmekh_lib.utils.xml import append_value_element, render_xml


def known_cost(trial: Trial) -> int:
    cost = trial.total_tokens
    assert cost is not None
    return cost


def quality(trial: Trial) -> float:
    score = trial.assessment.quality
    assert score is not None
    return score


def variant_summary(rows: list[Trial], settings: CampaignSettings) -> dict[str, Any]:
    scored = [trial for trial in rows if trial.assessment.quality is not None]
    costed = [trial for trial in rows if trial.total_tokens is not None]
    scored_costed = [trial for trial in scored if trial.total_tokens is not None]
    tokens = sum(known_cost(trial) for trial in costed)
    successes = sum(trial.assessment.success is True for trial in costed)
    actor_costed = [
        trial for trial in rows if trial.actor_usage["total_tokens"] is not None
    ]
    judge_costed = [
        trial for trial in rows if trial.judge_usage["total_tokens"] is not None
    ]
    return {
        "attempts": len(rows),
        "scored": len(scored),
        "infrastructure_errors": sum(
            trial.infrastructure_error is not None for trial in rows
        ),
        "mean_quality": fmean(quality(trial) for trial in scored) if scored else None,
        "successes": sum(trial.assessment.success is True for trial in rows),
        "quality_stddev": stdev(quality(trial) for trial in scored)
        if len(scored) > 1
        else None,
        "complete_usage_trials": len(costed),
        "complete_actor_usage_trials": len(actor_costed),
        "complete_judge_usage_trials": len(judge_costed),
        "total_tokens": tokens if len(costed) == len(rows) else None,
        "known_total_tokens": sum(
            trial.actor_usage["known_total_tokens"]
            + trial.judge_usage["known_total_tokens"]
            for trial in rows
        ),
        "mean_actor_tokens": fmean(
            trial.actor_usage["total_tokens"] for trial in actor_costed
        )
        if actor_costed
        else None,
        "mean_judge_tokens": fmean(
            trial.judge_usage["total_tokens"] for trial in judge_costed
        )
        if judge_costed
        else None,
        "mean_total_tokens": tokens / len(costed) if costed else None,
        "tokens_per_success": tokens / successes
        if successes and len(costed) == len(rows)
        else None,
        "mean_cost_adjusted_quality": fmean(
            quality(trial) - settings.token_penalty_per_1000 * known_cost(trial) / 1000
            for trial in scored_costed
        )
        if scored_costed
        else None,
        "compaction_calls": sum(
            trial.actor_usage["compaction_calls"] for trial in rows
        ),
    }


def summary(trials: list[Trial], settings: CampaignSettings) -> dict[str, Any]:
    variants = sorted({trial.variant for trial in trials})
    return {
        "variants": {
            variant: variant_summary(
                [trial for trial in trials if trial.variant == variant], settings
            )
            for variant in variants
        },
        "paired": paired_summary(trials, variants),
        "by_split": split_summary(trials, variants),
    }


def split_summary(trials: list[Trial], variants: list[str]) -> dict[str, Any]:
    result = {}
    for split in sorted({trial.split for trial in trials}):
        result[split] = {}
        for variant in variants:
            rows = [
                trial
                for trial in trials
                if trial.split == split
                and trial.variant == variant
                and trial.assessment.quality is not None
            ]
            result[split][variant] = {
                "scored": len(rows),
                "mean_quality": fmean(quality(trial) for trial in rows)
                if rows
                else None,
            }
    return result


def paired_summary(trials: list[Trial], variants: list[str]) -> list[dict[str, Any]]:
    result = []
    for index, left in enumerate(variants):
        for right in variants[index + 1 :]:
            pairs = {}
            for trial in trials:
                if trial.variant in (left, right):
                    pairs.setdefault((trial.task, trial.repetition), {})[
                        trial.variant
                    ] = trial
            complete = [
                (pair[left], pair[right])
                for pair in pairs.values()
                if left in pair
                and right in pair
                and pair[left].assessment.quality is not None
                and pair[right].assessment.quality is not None
            ]
            costed = [
                (a, b)
                for a, b in complete
                if a.total_tokens is not None and b.total_tokens is not None
            ]
            result.append(
                {
                    "left": left,
                    "right": right,
                    "pairs": len(complete),
                    "mean_quality_delta_right_minus_left": fmean(
                        quality(b) - quality(a) for a, b in complete
                    )
                    if complete
                    else None,
                    "cost_pairs": len(costed),
                    "quality_delta_stddev": stdev(
                        quality(b) - quality(a) for a, b in complete
                    )
                    if len(complete) > 1
                    else None,
                    "mean_token_delta_right_minus_left": fmean(
                        known_cost(b) - known_cost(a) for a, b in costed
                    )
                    if costed
                    else None,
                    "right_dominates": sum(
                        b.assessment.success is True
                        and a.assessment.success is True
                        and b.assessment.quality >= a.assessment.quality
                        and known_cost(b) <= known_cost(a)
                        and (
                            b.assessment.quality > a.assessment.quality
                            or known_cost(b) < known_cost(a)
                        )
                        for a, b in costed
                    ),
                }
            )
    return result


def write_report(
    settings: CampaignSettings,
    manifest: dict[str, Any],
    trials: list[Trial],
    complete: bool,
) -> None:
    root = ET.Element("agent-benchmark", {"version": "1"})
    append_value_element(root, "manifest", manifest)
    append_value_element(root, "complete", complete)
    append_value_element(root, "summary", summary(trials, settings))
    append_value_element(root, "trials", [asdict(trial) for trial in trials])
    temporary = settings.output.with_suffix(".xml.tmp")
    temporary.write_text(render_xml(root) + "\n", encoding="utf-8")
    temporary.replace(settings.output)


def write_trial(trial: Trial, path: Path) -> None:
    root = ET.Element("benchmark-trial")
    append_value_element(root, "result", asdict(trial))
    path.write_text(render_xml(root) + "\n", encoding="utf-8")


def render_progress(trial: Trial) -> str:
    root = ET.Element("benchmark-progress")
    append_value_element(
        root,
        "trial",
        {
            "task": trial.task,
            "variant": trial.variant,
            "quality": trial.assessment.quality,
            "success": trial.assessment.success,
            "actor_usage": trial.actor_usage,
            "judge_usage": trial.judge_usage,
            "error": trial.infrastructure_error,
        },
    )
    return render_xml(root)


def code_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for folder in (root / "evaluations", root / "src/exmekh_lib")
        for path in sorted(folder.rglob("*"))
        if path.is_file()
        and path.suffix in (".py", ".md", ".json")
        and "__pycache__" not in path.parts
        and "results" not in path.parts
    }
