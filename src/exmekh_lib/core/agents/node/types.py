from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from exmekh_lib.core.models.tool_message import ToolMessage
from exmekh_lib.core.models.tool_output_reference import ToolOutputReference
from exmekh_lib.core.models.user_message import UserMessageContent
from exmekh_lib.core.tool_calls.execution_manager import ExecutionSnapshot

type NodeUserMessage = UserMessageContent


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

    def execution_metadata(self) -> dict[str, Any] | None:
        """Describe execution consistently for persistence without exposing its form."""
        if self.execution is not None:
            return self.execution.status_info()
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
