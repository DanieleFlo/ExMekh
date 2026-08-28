from __future__ import annotations

from dataclasses import dataclass
from typing import Any

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
