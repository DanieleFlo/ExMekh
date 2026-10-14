"""Interpret tool return values without exposing wrapper details to the runtime."""

from typing import Any

from exmekh_lib.core.models.external_tool_output import ExternalToolOutput
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

    def external_output(self) -> ExternalToolOutput | None:
        return (
            self._result.external_output
            if isinstance(self._result, PreformattedToolResult)
            else None
        )

    def delivery_failed(self) -> bool:
        return (
            isinstance(self._result, PreformattedToolResult)
            and self._result.delivery_failed
        )

    def storage_value(self) -> Any:
        return None if self.external_output() is not None else self.value()
