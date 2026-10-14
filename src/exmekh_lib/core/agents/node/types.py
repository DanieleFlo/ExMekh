from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal

from exmekh_lib.core.models.external_tool_output import ExternalToolOutput
from exmekh_lib.core.models.tool_message import ToolMessage
from exmekh_lib.core.models.tool_output_reference import ToolOutputReference
from exmekh_lib.core.models.tool_result import ToolResult
from exmekh_lib.core.models.user_message import UserMessageContent
from exmekh_lib.core.tool_calls.execution_manager import ExecutionSnapshot
from exmekh_lib.core.tool_calls.output_runtime import ToolOutputDelivery

type NodeUserMessage = UserMessageContent


@dataclass(frozen=True, slots=True)
class ToolCallOutcome:
    value: Any
    execution: ExecutionSnapshot | None = None
    elapsed_seconds: float | None = None
    inspection_path: str | None = None
    failed: bool = False

    @property
    def delivery_failed(self) -> bool:
        return self.failed or ToolResult(self.value).delivery_failed()


@dataclass(frozen=True, slots=True)
class NodeExecutionSnapshot:
    depth: int
    skill_path: str
    history: list[dict[str, Any]]
    history_tree: list[dict[str, Any]]


@dataclass(frozen=True, slots=True)
class ToolExecutionResult:
    """Execution output and history update to be applied by the runtime."""

    response: ToolMessage
    raw_result: Any = None
    save_result: bool = False
    output_id: str | None = None
    cache_identity: CacheIdentity | None = None
    replace_history_from: int | None = None
    preserved_responses: tuple[ToolMessage, ...] = ()
    execution: ExecutionSnapshot | None = None
    elapsed_seconds: float | None = None
    used_tools: tuple[ToolOutputReference, ...] | None = None
    failed: bool = False
    inspection_path: str | None = None
    executed_arguments: dict[str, Any] | None = None
    external_output: ExternalToolOutput | None = None

    def with_delivery(
        self, delivery: ToolOutputDelivery, output_id: str
    ) -> ToolExecutionResult:
        raw_result = self.raw_result
        if delivery.failure is not None:
            raw_result = delivery.failure
        elif delivery.result.external_output is not None:
            raw_result = None
        elif delivery.execution is not None:
            raw_result = delivery.execution.record_result()
        return replace(
            self,
            response=self.response.with_result(delivery.result),
            raw_result=raw_result,
            output_id=output_id,
            execution=delivery.execution or self.execution,
            external_output=delivery.result.external_output,
            failed=self.failed or delivery.result.delivery_failed,
        )

    def execution_metadata(self) -> dict[str, Any] | None:
        """Describe execution consistently for persistence without exposing its form."""
        if self.execution is not None:
            info = self.execution.status_info()
            info["response_type"] = (
                "failed" if self.failed else self.execution.response_type
            )
            return info
        if self.elapsed_seconds is None:
            return {"status": "failed"} if self.failed else None
        return {
            "status": "failed" if self.failed else "completed",
            "elapsed_seconds": self.elapsed_seconds,
        }

    @property
    def is_error(self) -> bool:
        """Classify execution failures without interpreting business result values."""
        return self.failed or (
            self.execution is not None
            and self.execution.status in ("failed", "cancelled")
        )


@dataclass(frozen=True, slots=True)
class CacheIdentity:
    """Identity of the capability whose recoverable output was produced."""

    name: str
    source: Literal["Tool", "DeepSkill"]
    tool_call_id: str
    args: dict[str, Any]
    history_position: int
    thinking: str | None = None
