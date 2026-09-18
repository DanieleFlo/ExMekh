from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qs.core.models.tool_message import ToolMessage
from qs.core.models.user_message import (
    FilesUserMessage,
    ImagesUserMessage,
    TextsUserMessage,
)

type NodeUserMessage = str | TextsUserMessage | ImagesUserMessage | FilesUserMessage


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
    replace_history_from: int | None = None
    preserved_responses: tuple[ToolMessage, ...] = ()
