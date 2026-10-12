"""Typed protocol definitions; tool arguments remain ordinary JSON schema values."""

import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Protocol

from exmekh_lib.core.models.assistant_message import ToolCall
from exmekh_lib.core.models.user_message import UserMessage
from exmekh_lib.utils.xml import stringify_tool_result


class InteractiveProtocolError(RuntimeError):
    """An untrusted server response violates the negotiated contract."""


@dataclass(frozen=True, slots=True)
class InteractiveDefinition:
    behavior: str
    fields: Mapping[str, str]

    def configuration(self, name: str, schema: dict[str, Any]) -> dict[str, object]:
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        field = self.fields.get("document")
        if (
            self.behavior != "edit_file.v2"
            or set(self.fields) != {"document"}
            or set(properties) != {field}
            or set(required) != {field}
            or properties.get(field, {}).get("type") != "string"
        ):
            raise InteractiveProtocolError(
                "XML edit behavior requires exactly one required document string"
            )
        return {"name": name, "behavior": self.behavior, "fields": dict(self.fields)}


def validate_interactive_tool(
    interactive: InteractiveDefinition | None, *, background_enabled: bool
) -> None:
    """Apply the same interactive metadata policy to both tool definition APIs."""
    if interactive is None:
        return
    if not isinstance(interactive, InteractiveDefinition):
        raise TypeError("interactive must be an InteractiveDefinition or None")
    if background_enabled:
        raise ValueError("Interactive tools must execute synchronously")


class InteractiveHandler(Protocol):
    def begin(self) -> None: ...

    def prepare(
        self, event: dict[str, Any], definition: dict[str, Any]
    ) -> dict[str, object]: ...

    def validate_completed(self, calls: list[ToolCall]) -> None: ...

    def cancellation_requested(self) -> bool: ...


class InterruptedMessage(UserMessage):
    """Runtime evidence, never an executable call or an orphan tool result."""

    def __init__(self, record: dict[str, Any]) -> None:
        self._record = deepcopy(record)
        super().__init__(
            stringify_tool_result({"interactive_call_interrupted": self._record})
        )

    @property
    def record(self) -> dict[str, Any]:
        return deepcopy(self._record)

    def message_dump(self) -> dict[str, str]:
        return {
            **super().message_dump(),
            "interactive": json.dumps(self._record, ensure_ascii=True),
        }
