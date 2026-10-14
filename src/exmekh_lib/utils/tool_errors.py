"""Shared results for invocation failures and unavailable tool dependencies."""

import errno
import inspect
from copy import deepcopy
from typing import NotRequired, Self, TypedDict

from exmekh_lib.config.runtime import RUNTIME
from exmekh_lib.config.tool_errors import ERROR_HINTS


class ToolErrorDiagnostic(TypedDict):
    error_code: str
    hint: str
    details: NotRequired[dict[str, object]]


class ToolDiagnosticError(Exception):
    """A detected tool failure with a stable cause and recovery information."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "invalid_input",
        hint: str | None = None,
        details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self._code = code
        self._hint = (
            ERROR_HINTS.get(code, ERROR_HINTS["tool_execution_failed"])
            if hint is None
            else hint
        )
        self._details = deepcopy(details) if details is not None else {}

    @property
    def code(self) -> str:
        return self._code

    @property
    def hint(self) -> str:
        return self._hint

    @property
    def details(self) -> dict[str, object]:
        return deepcopy(self._details)

    def add_details(self, details: dict[str, object]) -> None:
        """Record observed context without retaining the caller's mutable objects."""
        self._details.update(deepcopy(details))

    def diagnostic_fields(self) -> ToolErrorDiagnostic:
        fields: ToolErrorDiagnostic = {"error_code": self.code, "hint": self.hint}
        if self._details:
            fields["details"] = self.details
        return fields


class ToolInputError(ToolDiagnosticError, ValueError):
    """Expected input or resource rejection, compatible with ValueError callers."""


class ToolTypeError(ToolDiagnosticError, TypeError):
    """Expected type rejection, compatible with TypeError callers."""


def error_diagnostic(error: BaseException) -> ToolErrorDiagnostic:
    """Describe detected failures without guessing from exception message text."""
    if isinstance(error, ToolDiagnosticError):
        return error.diagnostic_fields()
    code = "tool_execution_failed"
    if isinstance(error, FileNotFoundError):
        code = "file_not_found"
    elif isinstance(error, PermissionError):
        code = "permission_denied"
    elif isinstance(error, (IsADirectoryError, NotADirectoryError)):
        code = "not_a_regular_file"
    elif isinstance(error, UnicodeError):
        code = "invalid_encoding"
    elif isinstance(error, ZeroDivisionError):
        code = "undefined_math_result"
    elif isinstance(error, OSError):
        code = "disk_full" if error.errno == errno.ENOSPC else "filesystem_error"
    return {"error_code": code, "hint": ERROR_HINTS[code]}


def tool_error_result(error: BaseException, error_kind: str) -> dict[str, object]:
    """Keep the existing error family and add actionable cause information."""
    return {
        "ok": False,
        "error_kind": error_kind,
        "error": str(error),
        **error_diagnostic(error),
    }


def missing_dependency(error: ModuleNotFoundError) -> dict[str, object]:
    return {
        "ok": False,
        "error_kind": "missing_dependency",
        "error": RUNTIME.missing_dependency_message.format(module=error.name),
        "hint": RUNTIME.missing_dependency_hint,
    }


class ToolArgumentError(ToolDiagnosticError, TypeError):
    """Arguments rejected before invoking the tool, distinct from a tool failure."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "invalid_tool_arguments",
        hint: str | None = None,
        details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message, code=code, hint=hint, details=details)

    @classmethod
    def from_error(
        cls,
        error: BaseException,
        message: str,
        *,
        code: str = "invalid_tool_arguments",
        hint: str | None = None,
    ) -> Self:
        """Add invocation context while preserving an already detected cause."""
        if isinstance(error, ToolDiagnosticError):
            return cls(message, code=error.code, hint=error.hint, details=error.details)
        return cls(message, code=code, hint=hint)

    @classmethod
    def from_binding_error(
        cls,
        error: TypeError,
        signature: inspect.Signature,
        arguments: dict[str, object],
    ) -> Self:
        """Explain a binding failure from the same signature used to validate it."""
        parameters = signature.parameters
        required = [
            name
            for name, parameter in parameters.items()
            if parameter.default is inspect.Parameter.empty
            and parameter.kind
            not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        ]
        missing = [name for name in required if name not in arguments]
        accepts_keywords = any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
        unexpected = (
            [] if accepts_keywords else sorted(set(arguments) - parameters.keys())
        )
        if unexpected:
            code = "unexpected_tool_arguments"
        elif missing:
            code = "missing_tool_arguments"
        else:
            code = "invalid_tool_arguments"
        return cls(
            str(error),
            code=code,
            details={
                "missing": missing,
                "unexpected": unexpected,
                "required": required,
                "accepted": list(parameters),
            },
        )


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
        **error_diagnostic(error),
    }
