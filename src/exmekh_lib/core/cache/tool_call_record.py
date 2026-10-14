from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal, Protocol, Self, runtime_checkable

from exmekh_lib.config.cache_text import CACHE_TEXT
from exmekh_lib.core.models.assistant_message import ToolCall
from exmekh_lib.core.models.external_tool_output import ExternalToolOutput
from exmekh_lib.core.models.tool_output_reference import ToolOutputReference
from exmekh_lib.utils.json import to_json_value


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
    external_output: ExternalToolOutput | None = None

    def externalized(self, reference: ExternalToolOutput) -> ToolCallRecord:
        return replace(self, result=None, external_output=reference)

    def with_delivery_failure(self, failure: dict[str, Any]) -> ToolCallRecord:
        execution = {**(self.execution or {}), "response_type": "failed"}
        return replace(self, result=failure, external_output=None, execution=execution)

    def __post_init__(self) -> None:
        if self.external_output is not None:
            if not isinstance(self.external_output, ExternalToolOutput):
                raise TypeError(
                    CACHE_TEXT.external_output_must_be_externaltooloutput_or_none
                )
            if self.result is not None:
                raise ValueError(CACHE_TEXT.external_output_excludes_original_result)
        if not isinstance(self.name, str):
            raise TypeError(CACHE_TEXT.toolcallrecord_name_must_be_a_string)
        clean_name = self.name.strip()
        if not clean_name:
            raise ValueError(CACHE_TEXT.toolcallrecord_name_cannot_be_empty)
        object.__setattr__(self, "name", clean_name)

        if not isinstance(self.id, str):
            raise TypeError(CACHE_TEXT.toolcallrecord_id_must_be_a_string)
        clean_id = self.id.strip()
        if not clean_id:
            raise ValueError(CACHE_TEXT.toolcallrecord_id_cannot_be_empty)
        object.__setattr__(self, "id", clean_id)

        if self.tool_call_id is not None and (
            not isinstance(self.tool_call_id, str) or not self.tool_call_id.strip()
        ):
            raise ValueError(CACHE_TEXT.tool_call_id_must_be_a_non_empty_string)

        if self.thinking is not None and not isinstance(self.thinking, str):
            raise TypeError(CACHE_TEXT.thinking_must_be_a_string_or_none)

        if self.source not in ("Tool", "DeepSkill", "UserMessage"):
            raise ValueError(CACHE_TEXT.source_must_be_tool_deepskill_or_usermessage)

        if not isinstance(self.args, dict):
            raise TypeError(CACHE_TEXT.toolcallrecord_args_must_be_a_dictionary)

        object.__setattr__(self, "args", to_json_value(self.args))
        object.__setattr__(self, "result", to_json_value(self.result))
        if self.user_message is not None:
            if not isinstance(self.user_message, dict):
                raise TypeError(CACHE_TEXT.user_message_must_be_a_dictionary_or_none)
            object.__setattr__(self, "user_message", to_json_value(self.user_message))
        if self.execution is not None:
            if not isinstance(self.execution, dict):
                raise TypeError(CACHE_TEXT.execution_must_be_a_dictionary_or_none)
            object.__setattr__(self, "execution", to_json_value(self.execution))

        if not isinstance(self.skill_path, str):
            raise TypeError(CACHE_TEXT.skill_path_must_be_a_string)
        if not self.skill_path.startswith("/"):
            raise ValueError(CACHE_TEXT.skill_path_must_be_an_absolute_skill_path)

        if self.used_tools is not None and (
            not isinstance(self.used_tools, tuple)
            or not all(
                isinstance(item, ToolOutputReference) for item in self.used_tools
            )
        ):
            raise TypeError(
                CACHE_TEXT.used_tools_must_be_a_tuple_of_tooloutputreference_or
            )

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
            **(
                {"external_output": self.external_output.to_dict()}
                if self.external_output is not None
                else {}
            ),
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
            raise TypeError(CACHE_TEXT.record_dict_must_be_a_dictionary)
        required_keys = {"name", "id", "args", "result"}
        missing = required_keys - record_dict.keys()
        if missing:
            raise ValueError(
                CACHE_TEXT.missing_required_keys_in_tool_call_record.format(
                    missing=sorted(missing)
                )
            )
        used_tools = record_dict.get("used_tools")
        if used_tools is not None and not isinstance(used_tools, list):
            raise TypeError(CACHE_TEXT.serialized_used_tools_must_be_a_list_or_none)
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
            external_output=(
                ExternalToolOutput.from_dict(record_dict["external_output"])
                if "external_output" in record_dict
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
                CACHE_TEXT.tool_call_must_be_toolcall_or_implement_calledtoolmetadata
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
