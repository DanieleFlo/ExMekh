"""Contracts for tools invoked automatically by the harness."""

from dataclasses import dataclass
from typing import Any

from exmekh_lib.config.capability_prompts import CAPABILITY_PROMPTS
from exmekh_lib.config.default_capabilities import DEFAULT_CAPABILITIES
from exmekh_lib.core.file_inspection.evidence import InspectionReport
from exmekh_lib.core.models.tool import PreformattedToolResult, Tool


@dataclass(frozen=True, slots=True)
class ToolRuntimeContract:
    arguments: tuple[str, ...]
    result_type: type
    required_arguments: tuple[str, ...]

    def validate(self, tool: Tool) -> None:
        try:
            signature = tool.get_work_signature()
            signature.bind(**dict.fromkeys(self.arguments))
            signature.bind(**dict.fromkeys(self.required_arguments))
        except TypeError as error:
            raise ValueError(
                CAPABILITY_PROMPTS.invalid_contract.format(
                    name=tool.get_name(), arguments=", ".join(self.arguments)
                )
            ) from error

    def check_result(self, name: str, result: Any) -> Any:
        if (
            self.result_type is InspectionReport
            and isinstance(result, dict)
            and result.get("ok") is False
        ):
            return result
        if not isinstance(result, self.result_type):
            raise TypeError(
                CAPABILITY_PROMPTS.invalid_result.format(
                    name=name,
                    actual=type(result).__name__,
                    expected=self.result_type.__name__,
                )
            )
        return result


RUNTIME_CONTRACTS = {
    DEFAULT_CAPABILITIES.read_tool_output: ToolRuntimeContract(
        ("output_id", "cursor", "summary_mode"), PreformattedToolResult, ("output_id",)
    ),
    DEFAULT_CAPABILITIES.inspect_file: ToolRuntimeContract(
        ("path",), InspectionReport, ("path",)
    ),
}


def apply_runtime_contract(tool: Tool) -> Tool:
    contract = RUNTIME_CONTRACTS.get(tool.get_name())
    if contract is None:
        return tool
    contract.validate(tool)
    return tool.with_result_validator(
        lambda result: contract.check_result(tool.get_name(), result)
    )
