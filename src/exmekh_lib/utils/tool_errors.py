"""Shared results for invocation failures and unavailable tool dependencies."""


def missing_dependency(error: ModuleNotFoundError) -> dict[str, object]:
    return {
        "ok": False,
        "error_kind": "missing_dependency",
        "error": f"Missing exmekh_lib runtime module: {error.name}.",
        "hint": "Install the ExMekh-Lib wheel with its declared dependencies in this environment.",
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
        "error": f"{type(error).__name__}: {error}",
    }
