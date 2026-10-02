from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol, Self, runtime_checkable

from qs.core.models.assistant_message import ToolCall
from qs.core.models.tool_output_reference import ToolOutputReference
from qs.utils.json import to_json_value


@runtime_checkable
class CalledToolMetadata(Protocol):
    """Metadata shared by resolved tool, skill and exit calls."""

    @property
    def name(self) -> str: ...

    @property
    def call_id(self) -> str: ...

    @property
    def arguments(self) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class ToolCallRecord:
    """Tool output and its invocation metadata; field bindings are immutable."""

    name: str
    id: str
    args: dict[str, Any]
    result: Any
    skill_path: str = "/"
    tool_call_id: str | None = None
    thinking: str | None = None
    source: Literal["Tool", "DeepSkill", "UserMessage"] = "Tool"
    execution: dict[str, Any] | None = None
    used_tools: tuple[ToolOutputReference, ...] | None = None
    user_message: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError("ToolCallRecord 'name' must be a string")
        clean_name = self.name.strip()
        if not clean_name:
            raise ValueError("ToolCallRecord 'name' cannot be empty")
        object.__setattr__(self, "name", clean_name)

        if not isinstance(self.id, str):
            raise TypeError("ToolCallRecord 'id' must be a string")
        clean_id = self.id.strip()
        if not clean_id:
            raise ValueError("ToolCallRecord 'id' cannot be empty")
        object.__setattr__(self, "id", clean_id)

        if self.tool_call_id is not None and (
            not isinstance(self.tool_call_id, str) or not self.tool_call_id.strip()
        ):
            raise ValueError("tool_call_id must be a non-empty string or None")

        if self.thinking is not None and not isinstance(self.thinking, str):
            raise TypeError("thinking must be a string or None")

        if self.source not in ("Tool", "DeepSkill", "UserMessage"):
            raise ValueError("source must be 'Tool', 'DeepSkill', or 'UserMessage'")

        if not isinstance(self.args, dict):
            raise TypeError("ToolCallRecord 'args' must be a dictionary")

        object.__setattr__(self, "args", to_json_value(self.args))
        object.__setattr__(self, "result", to_json_value(self.result))
        if self.user_message is not None:
            if not isinstance(self.user_message, dict):
                raise TypeError("user_message must be a dictionary or None")
            object.__setattr__(self, "user_message", to_json_value(self.user_message))
        if self.execution is not None:
            if not isinstance(self.execution, dict):
                raise TypeError("execution must be a dictionary or None")
            object.__setattr__(self, "execution", to_json_value(self.execution))

        if not isinstance(self.skill_path, str):
            raise TypeError("skill_path must be a string")
        if not self.skill_path.startswith("/"):
            raise ValueError("skill_path must be an absolute skill path")

        if self.used_tools is not None and (
            not isinstance(self.used_tools, tuple)
            or not all(
                isinstance(item, ToolOutputReference) for item in self.used_tools
            )
        ):
            raise TypeError("used_tools must be a tuple of ToolOutputReference or None")

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dictionary."""
        return {
            "name": self.name,
            "id": self.id,
            "args": to_json_value(self.args),
            "result": to_json_value(self.result),
            "skill_path": self.skill_path,
            "tool_call_id": self.tool_call_id,
            "thinking": self.thinking,
            "type": self.source,
            **({"execution": self.execution} if self.execution is not None else {}),
            **(
                {"user_message": self.user_message}
                if self.user_message is not None
                else {}
            ),
            **(
                {"used_tools": [reference.to_dict() for reference in self.used_tools]}
                if self.used_tools is not None
                else {}
            ),
        }

    @classmethod
    def from_dict(cls, record_dict: dict[str, Any]) -> Self:
        """Construct a ToolCallRecord from a serialized dictionary."""
        if not isinstance(record_dict, dict):
            raise TypeError("record_dict must be a dictionary")
        required_keys = {"name", "id", "args", "result"}
        missing = required_keys - record_dict.keys()
        if missing:
            raise ValueError(
                f"Missing required keys in tool call record: {sorted(missing)}"
            )
        used_tools = record_dict.get("used_tools")
        if used_tools is not None and not isinstance(used_tools, list):
            raise TypeError("Serialized used_tools must be a list or None")
        return cls(
            name=record_dict["name"],
            id=record_dict["id"],
            args=record_dict["args"],
            result=record_dict["result"],
            skill_path=record_dict.get("skill_path", "/"),
            tool_call_id=record_dict.get("tool_call_id"),
            thinking=record_dict.get("thinking"),
            source=record_dict.get("type", "Tool"),
            execution=record_dict.get("execution"),
            user_message=record_dict.get("user_message"),
            used_tools=(
                tuple(ToolOutputReference.from_dict(item) for item in used_tools)
                if used_tools is not None
                else None
            ),
        )

    @classmethod
    def from_call(
        cls,
        tool_call: CalledToolMetadata | ToolCall,
        result: Any,
        id: str | None = None,
        *,
        skill_path: str = "/",
        thinking: str | None = None,
        source: Literal["Tool", "DeepSkill", "UserMessage"] = "Tool",
    ) -> Self:
        """Adapt the two supported call interfaces without guessing attributes."""
        if isinstance(tool_call, ToolCall):
            name = tool_call.get_name()
            call_id = tool_call.get_id()
            arguments = tool_call.get_arguments()
        elif isinstance(tool_call, CalledToolMetadata):
            name = tool_call.name
            call_id = tool_call.call_id
            arguments = tool_call.arguments
        else:
            raise TypeError(
                "tool_call must be ToolCall or implement CalledToolMetadata"
            )
        return cls(
            name=name,
            id=call_id if id is None else id,
            tool_call_id=call_id,
            args=arguments,
            result=result,
            skill_path=skill_path,
            thinking=thinking,
            source=source,
        )
