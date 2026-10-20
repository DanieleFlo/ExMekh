"""Versioned task contracts; grading policy is fixed before candidate execution."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ANALYTIC_WEIGHTS = {"exact": 1.0, "strong": 0.8, "partial": 0.35, "semantic": 0.0}


class Definition(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class Criterion(Definition):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    description: str = Field(min_length=1)
    importance: float = Field(default=1, gt=0)
    verification: Literal["exact", "strong", "partial", "semantic"]
    verification_reason: str = Field(min_length=1)
    checker: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    required: bool = False
    passing_score: float = Field(default=70, ge=0, le=100)

    @property
    def analytic_weight(self) -> float:
        return ANALYTIC_WEIGHTS[self.verification]

    @model_validator(mode="after")
    def require_checker(self):
        if self.analytic_weight and not self.checker:
            raise ValueError("Analytically weighted criteria require a checker")
        if not self.analytic_weight and self.checker:
            raise ValueError("Semantic criteria do not consume an analytical checker")
        return self


class Limits(Definition):
    max_calls: int = Field(default=12, gt=0)
    max_total_tokens: int = Field(default=100_000, gt=0)
    max_context_tokens: int = Field(default=24_000, gt=0)
    max_output_tokens: int = Field(default=2048, gt=0)
    max_seconds: float = Field(default=600, gt=0)
    compaction: Literal["forbid", "allow"] = "allow"
    compaction_threshold: float = Field(default=0.9, gt=0, le=1)
    compact_after_calls: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_limits(self):
        if self.max_output_tokens >= self.max_context_tokens:
            raise ValueError(
                "Output-token reserve must be smaller than context capacity"
            )
        if self.compact_after_calls and self.compaction == "forbid":
            raise ValueError("Scheduled compaction requires compaction='allow'")
        return self

    @property
    def prompt_limit(self) -> int:
        return self.max_context_tokens - self.max_output_tokens


class Task(Definition):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    domain: str
    split: Literal["development", "holdout"] = "development"
    driver: str = "native"
    prompt: str = Field(min_length=1)
    files: dict[str, str] = Field(default_factory=dict)
    artifacts: tuple[str, ...] = ()
    parameters: dict[str, Any] = Field(default_factory=dict)
    criteria: tuple[Criterion, ...] = Field(min_length=1)
    limits: Limits = Field(default_factory=Limits)

    @model_validator(mode="after")
    def unique_criteria(self):
        if len({item.id for item in self.criteria}) != len(self.criteria):
            raise ValueError("Criterion IDs must be unique within a task")
        return self


class Suite(Definition):
    version: Literal[1] = 1
    name: str
    tasks: tuple[Task, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_tasks(self):
        if len({item.id for item in self.tasks}) != len(self.tasks):
            raise ValueError("Task IDs must be unique within a suite")
        return self


class Variant(Definition):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    system_addendum: str = ""
    parameter_descriptions_in_schema: bool = False


@dataclass(frozen=True)
class Check:
    score: float | None
    evidence: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass(frozen=True)
class Observation:
    status: str
    answer: str = ""
    artifacts: dict[str, Any] = field(default_factory=dict)
    calls: tuple[dict[str, Any], ...] = ()
    events: tuple[dict[str, Any], ...] = ()
    error: str | None = None


@dataclass(frozen=True)
class JudgeMark:
    criterion: str
    score: float
    evidence: tuple[str, ...]
    rationale: str


@dataclass(frozen=True)
class Assessment:
    quality: float | None
    success: bool | None
    criteria: tuple[dict[str, Any], ...]
    error: str | None = None


@dataclass(frozen=True)
class CampaignSettings:
    output: Path
    repetitions: int = 1
    seed: int = 20261020
    token_penalty_per_1000: float = 1
    actor_model: str = "unspecified"
    judge_model: str = "unspecified"
    actor_reasoning: str = "unspecified"
    judge_reasoning: str = "unspecified"
    provider_timeout_seconds: float | None = None
    plugins: tuple[str, ...] = ()
    judge_limits: Limits = field(
        default_factory=lambda: Limits(
            max_calls=5,
            max_total_tokens=80_000,
            max_context_tokens=24_000,
            max_output_tokens=4096,
        )
    )


@dataclass(frozen=True)
class Trial:
    task: str
    domain: str
    split: str
    variant: str
    repetition: int
    observation: Observation
    assessment: Assessment
    actor_usage: dict[str, Any]
    judge_usage: dict[str, Any]
    elapsed_seconds: float
    infrastructure_error: str | None = None
    directory: str | None = None

    @property
    def total_tokens(self) -> int | None:
        actor = self.actor_usage["total_tokens"]
        judge = self.judge_usage["total_tokens"]
        return actor + judge if actor is not None and judge is not None else None
