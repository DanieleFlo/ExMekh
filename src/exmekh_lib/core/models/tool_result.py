"""Interpret tool return values without exposing wrapper details to the runtime."""

from typing import Any

from exmekh_lib.core.models.tool import AttachedToolResult, PreformattedToolResult
from exmekh_lib.core.models.user_message import UserMessage


class ToolResult:
    """Adapt ordinary, preformatted and attached results to the same public interface."""

    def __init__(self, result: Any) -> None:
        self._result = result

    def value(self) -> Any:
        """Keep structured values intact and unwrap tool-authored text."""
        return (
            self._result.content
            if isinstance(self._result, PreformattedToolResult)
            else self._result
        )

    def attachments(self) -> UserMessage | None:
        """Expose supported companions through one interface for every execution mode."""
        return (
            self._result.attachments
            if isinstance(self._result, (AttachedToolResult, PreformattedToolResult))
            else None
        )
