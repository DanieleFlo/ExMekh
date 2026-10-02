from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from exmekh_lib.interfaces.tool_hook import AfterTool, BeforeTool, validate_tool_hooks


class _UnchangedHook(Enum):
    VALUE = "unchanged"


@dataclass(frozen=True, slots=True)
class ToolHooks:
    """Omitted hooks stay unchanged; explicit None removes a hook."""

    before: BeforeTool | None | _UnchangedHook = _UnchangedHook.VALUE
    after: AfterTool | None | _UnchangedHook = _UnchangedHook.VALUE

    def __post_init__(self) -> None:
        validate_tool_hooks(
            None if isinstance(self.before, _UnchangedHook) else self.before,
            None if isinstance(self.after, _UnchangedHook) else self.after,
        )

    def resolve(
        self, before: BeforeTool | None, after: AfterTool | None
    ) -> tuple[BeforeTool | None, AfterTool | None]:
        return (
            before if isinstance(self.before, _UnchangedHook) else self.before,
            after if isinstance(self.after, _UnchangedHook) else self.after,
        )
