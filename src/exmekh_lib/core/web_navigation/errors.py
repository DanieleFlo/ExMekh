"""Errors owned by the web-navigation capability implementation."""

from typing import Self

from exmekh_lib.utils.tool_errors import ToolDiagnosticError


class WebNavigationError(ToolDiagnosticError, RuntimeError):
    """Describe one safe, model-facing web navigation failure."""

    def __init__(
        self,
        message: str,
        *,
        kind: str = "web_navigation",
        code: str | None = None,
        details: dict[str, object] | None = None,
    ) -> None:
        """Store a stable error kind and a concise public message."""
        super().__init__(message, code=code or kind, details=details)
        self.kind = kind

    @classmethod
    def from_crawl_failure(
        cls, message: str, *, status: object, requested_url: str
    ) -> Self:
        """Classify an observed HTTP status without inspecting provider text."""
        code = "crawl_failed"
        if type(status) is int:
            code = {
                404: "http_not_found",
                401: "http_access_denied",
                403: "http_access_denied",
                429: "http_rate_limited",
            }.get(status, "http_server_error" if 500 <= status < 600 else code)
        return cls(
            message,
            kind="crawl_failed",
            code=code,
            details={"status_code": status, "requested_url": requested_url},
        )
