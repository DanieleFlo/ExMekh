"""Shared results for invocation failures and unavailable tool dependencies."""

from exmekh_lib.config.runtime import RUNTIME


def missing_dependency(error: ModuleNotFoundError) -> dict[str, object]:
    return {
        "ok": False,
        "error_kind": "missing_dependency",
        "error": RUNTIME.missing_dependency_message.format(module=error.name),
        "hint": RUNTIME.missing_dependency_hint,
    }


class ToolArgumentError(TypeError):
    """Arguments rejected before invoking the tool, distinct from a tool failure."""


def tool_failure_result(error: BaseException) -> dict[str, object]:
    return {
        "ok": False,
        "error_kind": (
            "invalid_tool_arguments"
            if isinstance(error, ToolArgumentError)
            else "tool_execution_failed"
        ),
        "error": RUNTIME.failure_message.format(
            error_type=type(error).__name__, error=error
        ),
    }
